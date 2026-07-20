#include "loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

Model load_model(const char* filepath) {
    Model m = {0};
    m.fd = open(filepath, O_RDONLY);
    if (m.fd < 0) {
        printf("Error: Could not open file %s\n", filepath);
        exit(1);
    }
    
    struct stat sb;
    fstat(m.fd, &sb);
    m.file_size = sb.st_size;

    m.data = (unsigned char*)mmap(NULL, m.file_size, PROT_READ, MAP_PRIVATE, m.fd, 0);
    if (m.data == MAP_FAILED) {
        printf("Error: Memory mapping failed.\n");
        close(m.fd);
        exit(1);
    }

    m.config = (Config*)m.data;
    Config* c = m.config;
    
    // Start pointing right after the Config header
    float* ptr = (float*)(m.data + sizeof(Config));
    m.raw_weights = ptr;

    // --- MAP THE POINTERS ---
    int head_size = c->dim / c->n_heads;

    m.weights.token_embedding_table = ptr;
    ptr += c->vocab_size * c->dim;

    m.weights.rms_att_weight = ptr;
    ptr += c->n_layers * c->dim;

    m.weights.wq = ptr;
    ptr += c->n_layers * c->dim * c->dim;

    m.weights.wk = ptr;
    ptr += c->n_layers * c->dim * (c->n_kv_heads * head_size);

    m.weights.wv = ptr;
    ptr += c->n_layers * c->dim * (c->n_kv_heads * head_size);

    m.weights.wo = ptr;
    ptr += c->n_layers * c->dim * c->dim;

    m.weights.rms_ffn_weight = ptr;
    ptr += c->n_layers * c->dim;

    m.weights.w1 = ptr;
    ptr += c->n_layers * c->dim * c->hidden_dim;

    m.weights.w2 = ptr;
    ptr += c->n_layers * c->hidden_dim * c->dim;

    m.weights.w3 = ptr;
    ptr += c->n_layers * c->dim * c->hidden_dim;

    m.weights.rms_final_weight = ptr;
    ptr += c->dim;

    // Skip RoPE (Rotary Positional Embeddings) frequencies
    ptr += c->seq_len * head_size / 2; // real
    ptr += c->seq_len * head_size / 2; // imaginary

    // The classifier weights usually share the embedding table to save space.
    // We check if there are weights left in the file to see if they are separate.
    int file_has_classifier = ((unsigned char*)ptr < m.data + m.file_size);
    m.weights.wcls = file_has_classifier ? ptr : m.weights.token_embedding_table;

    return m;
}

void free_model(Model* model) {
    if (model->data && model->data != MAP_FAILED) {
        munmap(model->data, model->file_size);
    }
    if (model->fd >= 0) {
        close(model->fd);
    }
}