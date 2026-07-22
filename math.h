#ifndef MATH_H
#define MATH_H

void matmul(float* out, float* x, float* w, int n, int d);
void rmsnorm(float* out, float* x, float* weight, int size);
void forward_ffn(float* out, float* x, float* w1, float* w2, float* w3, int dim, int hidden_dim);
void rope(float* q, float* k, int pos, int head_size, int dim, int kv_dim);
void softmax(float* x, int size);
int argmax(float* logits, int size);
int sample_logits(float* logits, int size, float temperature);

#endif