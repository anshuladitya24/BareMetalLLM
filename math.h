#ifndef MATH_H
#define MATH_H

void matmul(float* out, float* x, float* w, int n, int d);
void silu(float* x, int size);
void rmsnorm(float* out, float* x, float* weight, int size);

#endif