// 这是C版本的chat.py，移植自chat.py，需要先python3 export_tokenizer.py导出gpt2_ranks.bin
// 该程序实现了一个简单的命令行界面，用户输入文本后会被编码成token ID，并通过管道传递给gpt程序进行推理，gpt程序输出的token ID会被解码回文本并显示出来。
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <limits.h>
#include <fcntl.h>

#define HASH_SIZE 131072 // 2^17
#define MAX_TOKENS 2048

// --- BPE 核心数据结构 ---
typedef struct {
    unsigned char* data;
    int len;
    int rank;
} RankItem;

RankItem* hash_table[HASH_SIZE];

unsigned int hash_bytes(unsigned char* data, int len) {
    unsigned int hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash % HASH_SIZE;
}

void load_ranks(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("Rank file missing"); exit(1); }
    unsigned char len;
    while (fread(&len, 1, 1, f) == 1) {
        RankItem* item = malloc(sizeof(RankItem));
        item->len = len;
        item->data = malloc(len);
        if (fread(item->data, 1, len, f) != (size_t)len) {
            fprintf(stderr, "Error reading rank data\n");
            exit(1);
        }
        if (fread(&item->rank, 4, 1, f) != 1) {
            fprintf(stderr, "Error reading rank value\n");
            exit(1);
        }
        unsigned int h = hash_bytes(item->data, len);
        while (hash_table[h]) h = (h + 1) % HASH_SIZE;
        hash_table[h] = item;
    }
    fclose(f);
}

int get_rank(unsigned char* data, int len) {
    unsigned int h = hash_bytes(data, len);
    while (hash_table[h]) {
        if (hash_table[h]->len == len && memcmp(hash_table[h]->data, data, len) == 0)
            return hash_table[h]->rank;
        h = (h + 1) % HASH_SIZE;
    }
    return INT_MAX;
}

// --- BPE 合并逻辑 ---
typedef struct {
    unsigned char* bytes;
    int len;
} Token;

int bpe_encode(const char* text, int* out_tokens) {
    int n = strlen(text);
    Token* tokens = malloc(sizeof(Token) * n);
    for (int i = 0; i < n; i++) {
        tokens[i].bytes = malloc(1);
        tokens[i].bytes[0] = (unsigned char)text[i];
        tokens[i].len = 1;
    }

    int current_count = n;
    while (current_count > 1) {
        int min_rank = INT_MAX;
        int best_idx = -1;

        for (int i = 0; i < current_count - 1; i++) {
            int combined_len = tokens[i].len + tokens[i+1].len;
            unsigned char* buf = malloc(combined_len);
            memcpy(buf, tokens[i].bytes, tokens[i].len);
            memcpy(buf + tokens[i].len, tokens[i+1].bytes, tokens[i+1].len);
            
            int r = get_rank(buf, combined_len);
            if (r < min_rank) {
                min_rank = r;
                best_idx = i;
            }
            free(buf);
        }

        if (best_idx == -1) break;

        // 执行合并
        int new_len = tokens[best_idx].len + tokens[best_idx+1].len;
        unsigned char* new_data = malloc(new_len);
        memcpy(new_data, tokens[best_idx].bytes, tokens[best_idx].len);
        memcpy(new_data + tokens[best_idx].len, tokens[best_idx+1].bytes, tokens[best_idx+1].len);
        
        free(tokens[best_idx].bytes);
        free(tokens[best_idx+1].bytes);
        tokens[best_idx].bytes = new_data;
        tokens[best_idx].len = new_len;

        for (int i = best_idx + 1; i < current_count - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        current_count--;
    }

    for (int i = 0; i < current_count; i++) {
        out_tokens[i] = get_rank(tokens[i].bytes, tokens[i].len);
        free(tokens[i].bytes);
    }
    free(tokens);
    return current_count;
}

// --- 主程序与进程通信 ---
int main() {
    load_ranks("models/gpt2_ranks.bin");

    char text[1024];
    printf("Text to complete: ");
    if (!fgets(text, sizeof(text), stdin)) return 0;
    text[strcspn(text, "\n")] = 0;

    int token_ids[MAX_TOKENS];
    int num = bpe_encode(text, token_ids);

    char* args[MAX_TOKENS + 2];
    // 修复路径问题，使用相对路径或绝对路径
    args[0] = "./gpt";  // 当前目录下
    for (int i = 0; i < num; i++) {
        args[i+1] = malloc(16);
        sprintf(args[i+1], "%d", token_ids[i]);
    }
    args[num+1] = NULL;

    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("Pipe failed");
        exit(1);
    }
    
    pid_t pid = fork();
    if (pid == 0) {
        // 子进程中
        close(pipefd[0]);  // 关闭读端
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);  // 关闭原写端
        
        // 尝试执行gpt程序
        execvp(args[0], args);
        perror("Execvp failed");
        exit(1);
    } else if (pid > 0) {
        // 父进程中
        close(pipefd[1]);  // 关闭写端
        FILE* out = fdopen(pipefd[0], "r");
        char line[32];
        while (fgets(line, sizeof(line), out)) {
            int tid = atoi(line);
            // 这里简化了解码逻辑：直接根据 Rank 查找对应的字节块
            // 在实际应用中，可复用 hash_table 来寻找 tid 对应的 bytes
            for(int i=0; i<HASH_SIZE; i++) {
                if(hash_table[i] && hash_table[i]->rank == tid) {
                    fwrite(hash_table[i]->data, 1, hash_table[i]->len, stdout);
                    fflush(stdout);
                    break;
                }
            }
        }
        fclose(out);
        wait(NULL);  // 等待子进程结束
    } else {
        perror("Fork failed");
        exit(1);
    }

    // 释放分配的内存
    for (int i = 0; i < num; i++) {
        free(args[i+1]);
    }
    
    return 0;
}