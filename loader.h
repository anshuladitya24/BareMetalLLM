#ifndef LOADER_H
#define LOADER_H

#include <stdint.h>
#include <stddef.h>

// The DNA of the model
typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

// Pointers to specific matrices inside the giant memory block
typedef struct {
    float* token_embedding_table; // (vocab_size, dim)
    
    // Attention Weights
    float* rms_att_weight;        // (layer, dim)
    float* wq;                    // (layer, dim, dim)
    float* wk;                    // (layer, dim, kv_dim)
    float* wv;                    // (layer, dim, kv_dim)
    float* wo;                    // (layer, dim, dim)
    
    // Feed Forward Weights
    float* rms_ffn_weight;        // (layer, dim)
    float* w1;                    // (layer, hidden_dim, dim)
    float* w2;                    // (layer, dim, hidden_dim)
    float* w3;                    // (layer, hidden_dim, dim)
    
    // Final Classifier
    float* rms_final_weight;      // (dim)
    float* wcls;                  // (vocab_size, dim)
} TransformerWeights;

// A container for the loaded model state
typedef struct {
    Config* config;
    TransformerWeights weights;
    float* raw_weights; // The pointer to the start of all weights
    unsigned char* data; 
    size_t file_size;
    int fd;
} Model;

Model load_model(const char* filepath);
void free_model(Model* model);

#endif