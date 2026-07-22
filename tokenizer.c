#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    
    // Allocate memory for the arrays
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        printf("Error: Could not open %s\n", tokenizer_path);
        exit(1);
    }

    // Read max token length
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        printf("Error: failed to read max_token_length\n");
        exit(1);
    }

    // Loop through the vocabulary and load each string
    int len;
    for (int i = 0; i < vocab_size; i++) {
        // Read the token score (used for encoding, not strictly needed for decoding)
        fread(&t->vocab_scores[i], sizeof(float), 1, file);
        
        // Read the string length
        fread(&len, sizeof(int), 1, file);
        
        // Allocate memory for the string (+1 for the null terminator)
        t->vocab[i] = (char*)malloc(len + 1);
        
        // Read the actual string
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0'; // Null terminate the string
    }
    
    fclose(file);
    printf("Successfully loaded vocabulary from %s\n", tokenizer_path);
}

char* decode(Tokenizer* t, int id) {
    if (id >= t->vocab_size || id < 0) {
        return "";
    }
    return t->vocab[id];
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
}

void encode(Tokenizer* t, char* text, int* tokens, int* n_tokens) {
    *n_tokens = 0;
    tokens[(*n_tokens)++] = 1; // Always start with the <BOS> token
    
    int len = strlen(text);
    for (int i = 0; i < len; ) {
        int best_id = -1;
        int best_len = 0;
        
        // Search the dictionary for the longest word that matches our text
        for (int id = 0; id < t->vocab_size; id++) {
            char* word = t->vocab[id];
            int w_len = strlen(word);
            // If the dictionary word matches the text at our current position
            if (w_len > 0 && strncmp(text + i, word, w_len) == 0) {
                if (w_len > best_len) {
                    best_len = w_len;
                    best_id = id;
                }
            }
        }
        
        if (best_id != -1) {
            tokens[(*n_tokens)++] = best_id;
            i += best_len; // Move forward by the length of the word we found
        } else {
            i++; // Fallback: skip unknown characters so we don't get stuck
        }
    }
}