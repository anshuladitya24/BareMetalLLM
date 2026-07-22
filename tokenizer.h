#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned int max_token_length;
} Tokenizer;

// Initialize the tokenizer by reading the tokenizer.bin file
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);

// Free the memory when we are done
void free_tokenizer(Tokenizer* t);

// Get the string associated with a Token ID
char* decode(Tokenizer* t, int id);

// Get the string associated with a Token ID
char* decode(Tokenizer* t, int id);

// NEW: Convert an English string into an array of Token IDs
void encode(Tokenizer* t, char* text, int* tokens, int* n_tokens);

#endif