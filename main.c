#include <stdio.h>
#include <stdlib.h>
#include <time.h>     // For our random seed
#include <math.h>     // C's standard math library (fixes sqrtf)
#include "loader.h"
#include "math.h"     // Our custom math library

int main(int argc, char *argv[])
{
    srand(time(NULL));
    if (argc < 2)
    {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    // 1. Load model and config
    Model model = load_model(argv[1]);
    Config *conf = model.config;

    printf("--- Model Loaded ---\n");
    printf("Dim: %d | Hidden Dim: %d | Layers: %d\n", conf->dim, conf->hidden_dim, conf->n_layers);

    // 2. Allocate buffers
    // [FIXED] Restored missing core vectors
    float *x = (float *)calloc(conf->dim, sizeof(float));
    float *x_normalized = (float *)calloc(conf->dim, sizeof(float));
    float *out = (float *)calloc(conf->dim, sizeof(float));

    int head_size = conf->dim / conf->n_heads;
    int kv_dim = conf->n_kv_heads * head_size;
    int kv_mul = conf->n_heads / conf->n_kv_heads;
    int pos = 0;
    int max_seq_len = 256;

    // Allocate temp vectors
    float *q = (float *)calloc(conf->dim, sizeof(float));
    float *k = (float *)calloc(kv_dim, sizeof(float));
    float *v = (float *)calloc(kv_dim, sizeof(float));
    float *att_out = (float *)calloc(conf->dim, sizeof(float));
    float *final_att = (float *)calloc(conf->dim, sizeof(float));

    // Allocate KV Cache for ALL layers
    float *key_cache = (float *)calloc(conf->n_layers * max_seq_len * kv_dim, sizeof(float));
    float *value_cache = (float *)calloc(conf->n_layers * max_seq_len * kv_dim, sizeof(float));

    // 3. Load initial token embedding (Token ID 1)
    // [FIXED] Restored the logic that injects the actual word into vector 'x'
    int token_id = 1;
    int embedding_offset = token_id * conf->dim;
    for (int i = 0; i < conf->dim; i++)
    {
        x[i] = model.weights.token_embedding_table[embedding_offset + i];
    }
    printf("Successfully loaded embedding for Token ID: %d\n", token_id);

    // --------------------------------------------------------
    // THE LAYER LOOP
    // --------------------------------------------------------
    printf("\n--- Entering the %d-Layer Transformer Stack ---\n", conf->n_layers);

    for (int l = 0; l < conf->n_layers; l++)
    {

        // --- POINTER MATH FOR CURRENT LAYER ---
        // We shift our pointers forward by (layer_index * size_of_one_layer)
        float *rms_att_layer = model.weights.rms_att_weight + l * conf->dim;
        float *rms_ffn_layer = model.weights.rms_ffn_weight + l * conf->dim;

        float *wq_layer = model.weights.wq + l * (conf->dim * conf->dim);
        float *wk_layer = model.weights.wk + l * (kv_dim * conf->dim);
        float *wv_layer = model.weights.wv + l * (kv_dim * conf->dim);
        float *wo_layer = model.weights.wo + l * (conf->dim * conf->dim);

        float *w1_layer = model.weights.w1 + l * (conf->dim * conf->hidden_dim);
        float *w2_layer = model.weights.w2 + l * (conf->hidden_dim * conf->dim);
        float *w3_layer = model.weights.w3 + l * (conf->dim * conf->hidden_dim);

        // Cache offset for this specific layer
        int layer_cache_offset = l * max_seq_len * kv_dim;

        // 1. Attention RMSNorm
        rmsnorm(x_normalized, x, rms_att_layer, conf->dim);

        // 2. Project Q, K, V
        matmul(q, x_normalized, wq_layer, conf->dim, conf->dim);
        matmul(k, x_normalized, wk_layer, kv_dim, conf->dim);
        matmul(v, x_normalized, wv_layer, kv_dim, conf->dim);

        // 3. Apply RoPE
        rope(q, k, pos, head_size, conf->dim, kv_dim);

        // 4. Store in KV Cache
        int current_cache_idx = layer_cache_offset + (pos * kv_dim);
        for (int i = 0; i < kv_dim; i++)
        {
            key_cache[current_cache_idx + i] = k[i];
            value_cache[current_cache_idx + i] = v[i];
        }

        // 5. Multi-Head Attention Math
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

        // 6. Output Projection & First Residual
        matmul(final_att, att_out, wo_layer, conf->dim, conf->dim);
        for (int i = 0; i < conf->dim; i++)
        {
            x[i] += final_att[i];
        }

        // 7. FFN RMSNorm
        rmsnorm(x_normalized, x, rms_ffn_layer, conf->dim);

        // 8. Feed-Forward Network & Second Residual
        forward_ffn(out, x_normalized, w1_layer, w2_layer, w3_layer, conf->dim, conf->hidden_dim);
        for (int i = 0; i < conf->dim; i++)
        {
            x[i] += out[i];
        }
    }
    printf("Successfully passed token through all %d layers.\n", conf->n_layers);

    // --------------------------------------------------------
    // FINAL CLASSIFIER (LOGITS)
    // --------------------------------------------------------
    // Final RMSNorm
    rmsnorm(x_normalized, x, model.weights.rms_final_weight, conf->dim);

    // The logits array holds a score for every single word in the vocabulary
    float *logits = (float *)calloc(conf->vocab_size, sizeof(float));

    // Multiply by the embedding table to get the final predictions
    // Note: In LLaMA, the final classifier uses the exact same weights as the token embedding!
    matmul(logits, x_normalized, model.weights.token_embedding_table, conf->vocab_size, conf->dim);

    // Set the creativity dial (1.0 is standard, 0.0 is strict greedy)
    float temperature = 0.9f;

    // Generate the next token!
    int next_token = sample_logits(logits, conf->vocab_size, temperature);

    printf("\n--- Final Generation ---\n");
    printf("The AI chose Token ID: %d\n", next_token);

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
    free(logits); // [FIXED] Actually freeing the logits array now
    free_model(&model);

    return 0;
}