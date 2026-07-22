#ifndef MATH_H
#define MATH_H

void matmul(float* out, float* x, float* w, int n, int d);
void silu(float* x, int size);
void rmsnorm(float* out, float* x, float* weight, int size);
void rope(float* q, float* k, int pos, int head_size, int dim, int kv_dim);
void softmax(float* x, int size);
#endif