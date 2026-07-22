#include "math.h"
#include <math.h>

// Matrix Multiplication: out = w * x
void matmul(float* out, float* x, float* w, int n, int d) {
    for (int i = 0; i < n; i++) {
        float val = 0.0f;
        for (int j = 0; j < d; j++) {
            val += w[i * d + j] * x[j];
        }
        out[i] = val;
    }
}

// Swish (SiLU) Activation: x = x * sigmoid(x)
void silu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

// We need a quick helper function to multiply vectors element-wise
void element_wise_mul(float* out, float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

// The SwiGLU Feed-Forward Network
void forward_ffn(float* out, float* x, float* w1, float* w2, float* w3, int dim, int hidden_dim) {
    // 1. We need temporary memory buffers for the two branches
    float* hb1 = (float*)calloc(hidden_dim, sizeof(float));
    float* hb2 = (float*)calloc(hidden_dim, sizeof(float));

    // 2. Branch 1: x * W1, then apply SiLU
    matmul(hb1, x, w1, hidden_dim, dim);
    silu(hb1, hidden_dim);

    // 3. Branch 2: x * W3
    matmul(hb2, x, w3, hidden_dim, dim);

    // 4. Multiply Branch 1 and Branch 2 together (hb1 = hb1 * hb2)
    element_wise_mul(hb1, hb1, hb2, hidden_dim);

    // 5. Final projection: hb1 * W2
    matmul(out, hb1, w2, dim, hidden_dim);

    // Free the temporary buffers
    free(hb1);
    free(hb2);
}

// Root Mean Square Normalization
void rmsnorm(float* out, float* x, float* weight, int size) {
    // 1. Calculate sum of squares
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    
    // 2. Find the mean and add epsilon for mathematical stability
    ss /= size;
    ss += 1e-5f; 
    ss = 1.0f / sqrtf(ss); // Inverse square root
    
    // 3. Normalize and apply the layer's specific RMS weights
    for (int i = 0; i < size; i++) {
        out[i] = weight[i] * (ss * x[i]);
    }
}

// RoPE: Rotary Positional Embedding
void rope(float* q, float* k, int pos, int head_size, int dim, int kv_dim) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        
        // Calculate the frequency (theta)
        float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)head_size);
        float val = (float)pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        // Rotate Query (Q)
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * fcr - q1 * fci;
        q[i + 1] = q0 * fci + q1 * fcr;

        // Rotate Key (K) - Only up to kv_dim since GQA has fewer key heads!
        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i]     = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
        }
    }

}

// Softmax: Turns raw scores into probabilities (0.0 to 1.0)
void softmax(float* x, int size) {
    // 1. Find max value for mathematical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // 2. Calculate the exponential of each element and the sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // 3. Normalize so they all sum to 1.0
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}