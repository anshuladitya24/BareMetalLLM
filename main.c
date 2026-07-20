#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "math.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    // 1. Load the Model 
    Model model = load_model(argv[1]);
    Config* conf = model.config;
    
    printf("--- Model Loaded ---\n");
    printf("Dim: %d | Hidden Dim: %d | Layers: %d\n", conf->dim, conf->hidden_dim, conf->n_layers);

    // 2. Prepare Buffers and Load Real Token
    float* x = (float*)calloc(conf->dim, sizeof(float)); 
    float* x_normalized = (float*)calloc(conf->dim, sizeof(float)); 
    float* out = (float*)calloc(conf->dim, sizeof(float));

    int token_id = 1; 
    int embedding_offset = token_id * conf->dim;

    for (int i = 0; i < conf->dim; i++) {
        x[i] = model.weights.token_embedding_table[embedding_offset + i];
    }
    
    printf("Successfully loaded embedding for Token ID: %d\n", token_id);

    // 3. Apply RMSNorm 
    // We use the specific RMS weights designed for the Feed-Forward Network
    rmsnorm(x_normalized, x, model.weights.rms_ffn_weight, conf->dim);
    printf("First float after RMSNorm: %f\n", x_normalized[0]);

    // 4. Execute the Full Feed-Forward Network Block
    printf("Executing SwiGLU FFN Block...\n");
    // Notice we pass 'x_normalized' here instead of 'x'!
    forward_ffn(out, x_normalized, model.weights.w1, model.weights.w2, model.weights.w3, conf->dim, conf->hidden_dim);
    
    printf("After FFN (First Val): %f\n", out[0]);

    // Cleanup
    free(x); free(x_normalized); free(out);
    free_model(&model);
    
    return 0;
}