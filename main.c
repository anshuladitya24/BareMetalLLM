#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "loader.h"
#include "math.h"
#include "tokenizer.h"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    // 1. Load model, config, and tokenizer
    Model model = load_model(argv[1]);
    Config *conf = model.config;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", conf->vocab_size);

    printf("--- Model Loaded ---\n");
    printf("Dim: %d | Hidden Dim: %d | Layers: %d\n", conf->dim, conf->hidden_dim, conf->n_layers);

    // 2. Allocate buffers
    float *x = (float *)calloc(conf->dim, sizeof(float));
    float *x_normalized = (float *)calloc(conf->dim, sizeof(float));
    float *out = (float *)calloc(conf->dim, sizeof(float));

    int head_size = conf->dim / conf->n_heads;
    int kv_dim = conf->n_kv_heads * head_size;
    int kv_mul = conf->n_heads / conf->n_kv_heads;
    int max_seq_len = 256;

    float *q = (float *)calloc(conf->dim, sizeof(float));
    float *k = (float *)calloc(kv_dim, sizeof(float));
    float *v = (float *)calloc(kv_dim, sizeof(float));
    float *att_out = (float *)calloc(conf->dim, sizeof(float));
    float *final_att = (float *)calloc(conf->dim, sizeof(float));
    float *key_cache = (float *)calloc(conf->n_layers * max_seq_len * kv_dim, sizeof(float));
    float *value_cache = (float *)calloc(conf->n_layers * max_seq_len * kv_dim, sizeof(float));
    float *logits = (float *)calloc(conf->vocab_size, sizeof(float));

    // 3. Setup Generation State
    char prompt_buffer[1024];
    char *prompt = NULL;

    if (argc >= 3)
    {
        // Option 1: Passed as a command-line argument
        prompt = argv[2];
    }
    else
    {
        // Option 2: Ask the user interactively in the program
        printf("\nEnter a prompt (or press Enter for a blank story):\n> "); // Added newline and arrow
        if (fgets(prompt_buffer, sizeof(prompt_buffer), stdin) != NULL)
        {
            size_t len = strlen(prompt_buffer);
            if (len > 0 && prompt_buffer[len - 1] == '\n')
            {
                prompt_buffer[len - 1] = '\0';
                len--;
            }
            if (len > 0)
            {
                prompt = prompt_buffer;
            }
        }
    }

    int prompt_tokens[256];
    int num_prompt_tokens = 0;

    // Ensure prompt starts with a space
    char formatted_prompt[1024];
    if (prompt != NULL && strlen(prompt) > 0)
    {
        if (prompt[0] != ' ')
        {
            snprintf(formatted_prompt, sizeof(formatted_prompt), " %s", prompt);
        }
        else
        {
            snprintf(formatted_prompt, sizeof(formatted_prompt), "%s", prompt);
        }

        // NEW: Convert normal ASCII spaces to LLaMA's special UTF-8 space character
        char safe_prompt[3072];
        int j = 0;
        for (int i = 0; formatted_prompt[i] != '\0'; i++)
        {
            if (formatted_prompt[i] == ' ')
            {
                safe_prompt[j++] = (char)0xE2; // LLaMA space byte 1
                safe_prompt[j++] = (char)0x96; // LLaMA space byte 2
                safe_prompt[j++] = (char)0x81; // LLaMA space byte 3
            }
            else
            {
                safe_prompt[j++] = formatted_prompt[i];
            }
        }
        safe_prompt[j] = '\0';

        // We pass the translated safe_prompt into the encoder
        encode(&tokenizer, safe_prompt, prompt_tokens, &num_prompt_tokens);

        printf("\n--- Generating Story ---\n");
        printf("%s", formatted_prompt);
        fflush(stdout);
    }
    else
    {
        prompt_tokens[0] = 1; // Just <BOS>
        num_prompt_tokens = 1;
        printf("\n--- Generating Story ---\n");
    }

    int next_token = prompt_tokens[0];
    float temperature = 1.0f;
    int generate_steps = 200;
    int pos = 0;

    // ========================================================
    // THE GENERATION LOOP
    // ========================================================
    for (pos = 0; pos < generate_steps; pos++)
    {

        // Inject current token embedding into x
        int embedding_offset = next_token * conf->dim;
        for (int i = 0; i < conf->dim; i++)
        {
            x[i] = model.weights.token_embedding_table[embedding_offset + i];
        }

        // Pass through Transformer Layers
        for (int l = 0; l < conf->n_layers; l++)
        {

            // Pointer Math for current layer
            float *rms_att_layer = model.weights.rms_att_weight + l * conf->dim;
            float *rms_ffn_layer = model.weights.rms_ffn_weight + l * conf->dim;

            float *wq_layer = model.weights.wq + l * (conf->dim * conf->dim);
            float *wk_layer = model.weights.wk + l * (kv_dim * conf->dim);
            float *wv_layer = model.weights.wv + l * (kv_dim * conf->dim);
            float *wo_layer = model.weights.wo + l * (conf->dim * conf->dim);

            float *w1_layer = model.weights.w1 + l * (conf->dim * conf->hidden_dim);
            float *w2_layer = model.weights.w2 + l * (conf->hidden_dim * conf->dim);
            float *w3_layer = model.weights.w3 + l * (conf->dim * conf->hidden_dim);

            int layer_cache_offset = l * max_seq_len * kv_dim;

            // Zero out attention output buffer before computing head values
            memset(att_out, 0, conf->dim * sizeof(float));

            // Attention RMSNorm & QKV Projection
            rmsnorm(x_normalized, x, rms_att_layer, conf->dim);
            matmul(q, x_normalized, wq_layer, conf->dim, conf->dim);
            matmul(k, x_normalized, wk_layer, kv_dim, conf->dim);
            matmul(v, x_normalized, wv_layer, kv_dim, conf->dim);

            // Apply RoPE & Cache K, V
            rope(q, k, pos, head_size, conf->dim, kv_dim);
            int current_cache_idx = layer_cache_offset + (pos * kv_dim);
            for (int i = 0; i < kv_dim; i++)
            {
                key_cache[current_cache_idx + i] = k[i];
                value_cache[current_cache_idx + i] = v[i];
            }

            // Multi-Head Attention Math
            for (int h = 0; h < conf->n_heads; h++)
            {
                float *q_head = q + h * head_size;
                float *att_scores = (float *)calloc(pos + 1, sizeof(float));
                int kv_head = h / kv_mul;

                for (int t = 0; t <= pos; t++)
                {
                    float *k_head = key_cache + layer_cache_offset + t * kv_dim + kv_head * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += q_head[i] * k_head[i];
                    }
                    att_scores[t] = score / sqrtf(head_size);
                }

                softmax(att_scores, pos + 1);

                float *out_head = att_out + h * head_size;
                for (int t = 0; t <= pos; t++)
                {
                    float *v_head = value_cache + layer_cache_offset + t * kv_dim + kv_head * head_size;
                    float score = att_scores[t];
                    for (int i = 0; i < head_size; i++)
                    {
                        out_head[i] += score * v_head[i];
                    }
                }
                free(att_scores);
            }

            // Attention Output & Residual
            matmul(final_att, att_out, wo_layer, conf->dim, conf->dim);
            for (int i = 0; i < conf->dim; i++)
            {
                x[i] += final_att[i];
            }

            // FFN RMSNorm, Execution, & Residual
            rmsnorm(x_normalized, x, rms_ffn_layer, conf->dim);
            forward_ffn(out, x_normalized, w1_layer, w2_layer, w3_layer, conf->dim, conf->hidden_dim);
            for (int i = 0; i < conf->dim; i++)
            {
                x[i] += out[i];
            }
        }

        // Final RMSNorm & Logits
        rmsnorm(x_normalized, x, model.weights.rms_final_weight, conf->dim);
        matmul(logits, x_normalized, model.weights.token_embedding_table, conf->vocab_size, conf->dim);

        // 5. Determine the next token (Prompt Forcing vs. Sampling)
        if (pos < num_prompt_tokens - 1)
        {
            // We are still processing the user's prompt!
            // Force the engine to use the exact word you typed.
            next_token = prompt_tokens[pos + 1];
        }
        else
        {
            // The prompt is finished. Let the AI take over and sample creatively.
            next_token = sample_logits(logits, conf->vocab_size, temperature);

            // Stop if EOS token (2) or BOS token (1) is produced
            if (next_token == 2 || next_token == 1)
            {
                break;
            }

            // Decode and display the AI's new words live
            char *word = decode(&tokenizer, next_token);

            if (strcmp(word, "<0x0A>") == 0)
            {
                printf("\n");
            }
            else
            {
                printf("%s", word);
            }
            fflush(stdout);
        }
    } // End of the generation loop

    printf("\n\n--- Generation Complete ---\n");

    // Cleanup
    free(x);
    free(x_normalized);
    free(out);
    free(q);
    free(k);
    free(v);
    free(key_cache);
    free(value_cache);
    free(att_out);
    free(final_att);
    free(logits);
    free_tokenizer(&tokenizer);
    free_model(&model);

    return 0;
}