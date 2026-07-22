#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "math.h"

int main(int argc, char *argv[])
{
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
    float *x = (float *)calloc(conf->dim, sizeof(float));
    float *x_normalized = (float *)calloc(conf->dim, sizeof(float));
    float *out = (float *)calloc(conf->dim, sizeof(float));

    // 3. Load initial token embedding (Token ID 1)
    int token_id = 1;
    int embedding_offset = token_id * conf->dim;

    for (int i = 0; i < conf->dim; i++)
    {
        x[i] = model.weights.token_embedding_table[embedding_offset + i];
    }

    printf("Successfully loaded embedding for Token ID: %d\n", token_id);

    // --------------------------------------------------------
    // 4. SELF-ATTENTION BLOCK
    // --------------------------------------------------------
    printf("\n--- Executing Self-Attention Block ---\n");

    // Apply RMSNorm for Attention
    rmsnorm(x_normalized, x, model.weights.rms_att_weight, conf->dim);

    int head_size = conf->dim / conf->n_heads;
    int kv_dim = conf->n_kv_heads * head_size;
    int kv_mul = conf->n_heads / conf->n_kv_heads;
    int pos = 0;

    // Allocate Attention vectors
    float *q = (float *)calloc(conf->dim, sizeof(float));
    float *k = (float *)calloc(kv_dim, sizeof(float));
    float *v = (float *)calloc(kv_dim, sizeof(float));

    // Allocate KV Cache (max sequence length 256)
    int max_seq_len = 256;
    float *key_cache = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    float *value_cache = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    float *att_out = (float *)calloc(conf->dim, sizeof(float));

    // Project input to Q, K, V
    matmul(q, x_normalized, model.weights.wq, conf->dim, conf->dim);
    matmul(k, x_normalized, model.weights.wk, kv_dim, conf->dim);
    matmul(v, x_normalized, model.weights.wv, kv_dim, conf->dim);

    // Apply RoPE
    rope(q, k, pos, head_size, conf->dim, kv_dim);

    // Store K and V in the cache
    int cache_offset = pos * kv_dim;
    for (int i = 0; i < kv_dim; i++)
    {
        key_cache[cache_offset + i] = k[i];
        value_cache[cache_offset + i] = v[i];
    }

    // Multi-Head Attention Math: Softmax(Q * K^T / sqrt(d)) * V
    for (int h = 0; h < conf->n_heads; h++)
    {
        float *q_head = q + h * head_size;
        float *att_scores = (float *)calloc(pos + 1, sizeof(float));
        int kv_head = h / kv_mul;

        // Matchmaking (Dot Product)
        for (int t = 0; t <= pos; t++)
        {
            float *k_head = key_cache + t * kv_dim + kv_head * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++)
            {
                score += q_head[i] * k_head[i];
            }
            att_scores[t] = score / sqrtf(head_size);
        }

        // Convert to probabilities
        softmax(att_scores, pos + 1);

        // Extract Value payload
        float *out_head = att_out + h * head_size;
        for (int t = 0; t <= pos; t++)
        {
            float *v_head = value_cache + t * kv_dim + kv_head * head_size;
            float score = att_scores[t];
            for (int i = 0; i < head_size; i++)
            {
                out_head[i] += score * v_head[i];
            }
        }
        free(att_scores);
    }

    // Final Output Projection (W_o)
    float *final_att = (float *)calloc(conf->dim, sizeof(float));
    matmul(final_att, att_out, model.weights.wo, conf->dim, conf->dim);

    printf("After Attention - First float: %f\n", final_att[0]);

    // Residual Connection 1: Add Attention output back to original token
    for (int i = 0; i < conf->dim; i++) {
        x[i] += final_att[i]; 
    }

    // --------------------------------------------------------
    // 5. FEED-FORWARD NETWORK BLOCK
    // --------------------------------------------------------
    printf("\n--- Executing SwiGLU FFN Block ---\n");

    // Apply RMSNorm for FFN
    rmsnorm(x_normalized, x, model.weights.rms_ffn_weight, conf->dim);

    // Execute Feed-Forward Network
    forward_ffn(out, x_normalized, model.weights.w1, model.weights.w2, model.weights.w3, conf->dim, conf->hidden_dim);
    
    // Residual Connection 2: Add FFN output back to token
    for (int i = 0; i < conf->dim; i++) {
        x[i] += out[i]; 
    }

    printf("After Layer 0 Forward Pass - First float of x: %f\n", x[0]);

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
    free_model(&model);

    return 0;
}