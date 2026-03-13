// Original Author: Andrej Karpathy
// https://github.com/karpathy/llm.c
//
// 所需文件:
//   gpt2_124M.bin   - GPT-2 模型文件
//   gpt2_ranks.bin   - BPE ranks 文件 (由 export_tokenizer.py 导出)

#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <limits.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <err.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>
/* 获取UUID */
#include <gpt_ta.h>
// ----------------------------------------------------------------------------
// BPE tokenizer 结构和函数 (from cchat.c)

#define HASH_SIZE 131072   // 2^17, must be larger than vocabulary size
#define MAX_TOKENS 2048    // maximum number of tokens we handle

/**
 * @struct RankItem
 * @brief 存储字节序列及其排名的结构体
 * @var RankItem::data
 * 字节序列数据
 * @var RankItem::len
 * 数据长度
 * @var RankItem::rank
 * 排名值
 */
typedef struct {
    unsigned char* data;
    int len;
    int rank;
} RankItem;

RankItem* hash_table[HASH_SIZE];

/**
 * @brief 计算字节序列的FNV-1a哈希值
 * @param data 字节序列指针
 * @param len 字节序列长度
 * @return 返回哈希值（在HASH_SIZE范围内）
 */
unsigned int hash_bytes(unsigned char* data, int len) {
    unsigned int hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash % HASH_SIZE;
}

/**
 * @brief 加载由export_tokenizer.py生成的ranks文件
 * @param path ranks文件路径
 * @return 无返回值，但会填充全局hash_table
 */
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

/**
 * @brief 获取字节序列的排名值
 * @param data 字节序列指针
 * @param len 字节序列长度
 * @return 返回排名值，如果未找到则返回INT_MAX
 */
int get_rank(unsigned char* data, int len) {
    unsigned int h = hash_bytes(data, len);
    while (hash_table[h]) {
        if (hash_table[h]->len == len && memcmp(hash_table[h]->data, data, len) == 0)
            return hash_table[h]->rank;
        h = (h + 1) % HASH_SIZE;
    }
    return INT_MAX;
}

/**
 * @struct Token
 * @brief 临时存储token的结构体，用于BPE合并
 * @var Token::bytes
 * 字节序列指针
 * @var Token::len
 * 长度
 */
typedef struct {
    unsigned char* bytes;
    int len;
} Token;

/**
 * @brief 使用BPE编码将UTF-8字符串编码为token ID序列
 * @param text 输入的UTF-8文本
 * @param[out] out_tokens 输出的token ID数组（假设已分配足够空间）
 * @return 返回token数量
 */
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

        // Merge the best pair
        int new_len = tokens[best_idx].len + tokens[best_idx+1].len;
        unsigned char* new_data = malloc(new_len);
        memcpy(new_data, tokens[best_idx].bytes, tokens[best_idx].len);
        memcpy(new_data + tokens[best_idx].len, tokens[best_idx+1].bytes, tokens[best_idx+1].len);
        
        free(tokens[best_idx].bytes);
        free(tokens[best_idx+1].bytes);
        tokens[best_idx].bytes = new_data;
        tokens[best_idx].len = new_len;

        // Shift remaining tokens left
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

// ----------------------------------------------------------------------------
// GPT-2 model definition and forward passes
// All tensor shapes are documented in the original code.

/**
 * @brief 编码器：token嵌入 + 位置嵌入
 * @param[out] out 输出张量，形状为(B, T, C)
 * @param inp 输入token IDs，形状为(B, T)
 * @param wte token嵌入权重，形状为(V, C)
 * @param wpe 位置嵌入权重，形状为(maxT, C)
 * @param B 批次大小
 * @param T 序列长度
 * @param C 通道数（隐藏层维度）
 */
void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* wte_ix = wte + ix * C;
            float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

/**
 * @brief 层归一化前向传播
 * @param[out] out 输出张量，形状为(B, T, C)
 * @param[out] mean 计算得到的均值，形状为(B, T)
 * @param[out] rstd 计算得到的标准差倒数，形状为(B, T)
 * @param inp 输入张量，形状为(B, T, C)
 * @param weight 归一化权重，形状为(C)
 * @param bias 归一化偏置，形状为(C)
 * @param B 批次大小
 * @param T 序列长度
 * @param C 通道数（隐藏层维度）
 */
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) m += x[i];
            m = m/C;
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            float s = 1.0f / sqrtf(v + eps);
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = s * (x[i] - m);
                out_bt[i] = n * weight[i] + bias[i];
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// Thread argument structure for matmul_forward
/**
 * @struct MatmulThreadArgs
 * @brief 矩阵乘法多线程计算参数结构体
 * @var MatmulThreadArgs::out
 * 输出矩阵指针
 * @var MatmulThreadArgs::inp
 * 输入矩阵指针
 * @var MatmulThreadArgs::weight
 * 权重矩阵指针
 * @var MatmulThreadArgs::bias
 * 偏置向量指针
 * @var MatmulThreadArgs::start
 * 起始索引
 * @var MatmulThreadArgs::end
 * 结束索引
 * @var MatmulThreadArgs::B
 * 批次大小
 * @var MatmulThreadArgs::T
 * 序列长度
 * @var MatmulThreadArgs::C
 * 输入通道数
 * @var MatmulThreadArgs::OC
 * 输出通道数
 */
typedef struct {
    float* out;
    float* inp, *weight, *bias;
    size_t start;
    size_t end;
    int B, T, C, OC;
} MatmulThreadArgs;

/**
 * @brief 矩阵乘法多线程计算工作函数
 * @param v 指向MatmulThreadArgs结构体的指针
 * @return 返回NULL
 */
static void* matmul_worker(void* v) {
    MatmulThreadArgs* a = (MatmulThreadArgs*)v;
    size_t BT_OC = (size_t)a->T * (size_t)a->OC;
    size_t BT_C = (size_t)a->T * (size_t)a->C;
    for (size_t index = a->start; index < a->end; index++) {
        size_t b = index / BT_OC;
        size_t rem1 = index % BT_OC;
        size_t t = rem1 / (size_t)a->OC;
        size_t o = rem1 % (size_t)a->OC;
        float* out_bt = a->out + b * BT_OC + t * a->OC;
        float* inp_bt = a->inp + b * BT_C + t * a->C;
        float val = (a->bias != NULL) ? a->bias[o] : 0.0f;
        float* wrow = a->weight + o * a->C;
        for (int i = 0; i < a->C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out_bt[o] = val;
    }
    return NULL;
}

#define nthreads 4

/**
 * @brief 矩阵乘法前向传播（多线程）
 * @param[out] out 输出张量，形状为(B, T, OC)
 * @param inp 输入张量，形状为(B, T, C)
 * @param weight 权重矩阵，形状为(C, OC)
 * @param bias 偏置向量，形状为(OC)，可为NULL
 * @param B 批次大小
 * @param T 序列长度
 * @param C 输入通道数
 * @param OC 输出通道数
 */
void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    size_t total = (size_t)B * (size_t)T * (size_t)OC;

    pthread_t* threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    MatmulThreadArgs* args = (MatmulThreadArgs*)malloc(nthreads * sizeof(MatmulThreadArgs));
    size_t base = total / nthreads;
    size_t rem = total % nthreads;

    for (int tid = 0; tid < nthreads; tid++) {
        size_t start = tid * base;
        size_t len = base + (tid == nthreads - 1 ? rem : 0);
        args[tid].out = out;
        args[tid].inp = inp;
        args[tid].weight = weight;
        args[tid].bias = bias;
        args[tid].start = start;
        args[tid].end = start + len;
        args[tid].B = B; args[tid].T = T; args[tid].C = C; args[tid].OC = OC;

        if (pthread_create(&threads[tid], NULL, matmul_worker, &args[tid]) != 0) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }

    for (int tid = 0; tid < nthreads; tid++) {
        if (pthread_join(threads[tid], NULL) != 0) {
            perror("pthread_join failed");
            exit(EXIT_FAILURE);
        }
    }

    free(threads);
    free(args);
}

/**
 * @brief 因果多头注意力机制前向传播
 * @param[out] out 输出张量，形状为(B, T, C)
 * @param[out] preatt 注意力预激活值，形状为(B, NH, T, T)
 * @param[out] att 注意力权重，形状为(B, NH, T, T)
 * @param inp 输入张量，形状为(B, T, 3*C)，包含QKV
 * @param B 批次大小
 * @param T 序列长度
 * @param C 总通道数
 * @param NH 头的数量
 */
void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    int C3 = C*3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) maxval = val;
                    preatt_bth[t2] = val;
                }

                // pass 2: exponentiate and sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: weighted sum of values
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

/**
 * @brief GELU激活函数（近似）
 * @param[out] out 输出张量，形状为(N)
 * @param inp 输入张量，形状为(N)
 * @param N 张量元素总数
 */
void gelu_forward(float* out, float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

/**
 * @brief 残差连接：out = inp1 + inp2
 * @param[out] out 输出张量，形状为(N)
 * @param inp1 第一个输入张量，形状为(N)
 * @param inp2 第二个输入张量，形状为(N)
 * @param N 张量元素总数
 */
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

/**
 * @brief 在词汇表维度上执行softmax操作
 * @param[out] probs 输出概率，形状为(B, T, V)
 * @param logits 输入逻辑值，形状为(B, T, V)
 * @param B 批次大小
 * @param T 序列长度
 * @param V 词汇表大小
 */
void softmax_forward(float* probs, float* logits, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            float maxval = -10000.0f;
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) maxval = logits_bt[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model structure (from gpt.c)

#define NUM_PARAMETER_TENSORS 16
/**
 * @struct ParameterTensors
 * @brief GPT-2模型参数张量结构体
 * @var ParameterTensors::wte
 * token嵌入权重，形状为(V, C)
 * @var ParameterTensors::wpe
 * 位置嵌入权重，形状为(maxT, C)
 * @var ParameterTensors::ln1w
 * 第一层归一化权重，形状为(L, C)
 * @var ParameterTensors::ln1b
 * 第一层归一化偏置，形状为(L, C)
 * @var ParameterTensors::qkvw
 * QKV变换权重，形状为(L, 3*C, C)
 * @var ParameterTensors::qkvb
 * QKV变换偏置，形状为(L, 3*C)
 * @var ParameterTensors::attprojw
 * 注意力投影权重，形状为(L, C, C)
 * @var ParameterTensors::attprojb
 * 注意力投影偏置，形状为(L, C)
 * @var ParameterTensors::ln2w
 * 第二层归一化权重，形状为(L, C)
 * @var ParameterTensors::ln2b
 * 第二层归一化偏置，形状为(L, C)
 * @var ParameterTensors::fcw
 * 前馈网络权重，形状为(L, 4*C, C)
 * @var ParameterTensors::fcb
 * 前馈网络偏置，形状为(L, 4*C)
 * @var ParameterTensors::fcprojw
 * 前馈网络投影权重，形状为(L, C, 4*C)
 * @var ParameterTensors::fcprojb
 * 前馈网络投影偏置，形状为(L, C)
 * @var ParameterTensors::lnfw
 * 最终层归一化权重，形状为(C)
 * @var ParameterTensors::lnfb
 * 最终层归一化偏置，形状为(C)
 */
typedef struct {
    float* wte;        // (V, C)
    float* wpe;        // (maxT, C)
    float* ln1w;       // (L, C)
    float* ln1b;       // (L, C)
    float* qkvw;       // (L, 3*C, C)
    float* qkvb;       // (L, 3*C)
    float* attprojw;   // (L, C, C)
    float* attprojb;   // (L, C)
    float* ln2w;       // (L, C)
    float* ln2b;       // (L, C)
    float* fcw;        // (L, 4*C, C)
    float* fcb;        // (L, 4*C)
    float* fcprojw;    // (L, C, 4*C)
    float* fcprojb;    // (L, C)
    float* lnfw;       // (C)
    float* lnfb;       // (C)
} ParameterTensors;

/**
 * @brief 分配并初始化参数张量内存
 * @param params 参数张量结构体指针
 * @param param_sizes 各参数张量大小数组
 * @return 返回分配的参数内存块
 */
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    float* params_memory = (float*)malloc(num_parameters * sizeof(float));
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
/**
 * @struct ActivationTensors
 * @brief GPT-2模型激活张量结构体
 * @var ActivationTensors::encoded
 * 编码输出，形状为(B, T, C)
 * @var ActivationTensors::ln1
 * 第一层归一化输出，形状为(L, B, T, C)
 * @var ActivationTensors::ln1_mean
 * 第一层归一化均值，形状为(L, B, T)
 * @var ActivationTensors::ln1_rstd
 * 第一层归一化标准差倒数，形状为(L, B, T)
 * @var ActivationTensors::qkv
 * QKV变换结果，形状为(L, B, T, 3*C)
 * @var ActivationTensors::atty
 * 注意力输出，形状为(L, B, T, C)
 * @var ActivationTensors::preatt
 * 注意力预激活值，形状为(L, B, NH, T, T)
 * @var ActivationTensors::att
 * 注意力权重，形状为(L, B, NH, T, T)
 * @var ActivationTensors::attproj
 * 注意力投影输出，形状为(L, B, T, C)
 * @var ActivationTensors::residual2
 * 第二个残差连接，形状为(L, B, T, C)
 * @var ActivationTensors::ln2
 * 第二层归一化输出，形状为(L, B, T, C)
 * @var ActivationTensors::ln2_mean
 * 第二层归一化均值，形状为(L, B, T)
 * @var ActivationTensors::ln2_rstd
 * 第二层归一化标准差倒数，形状为(L, B, T)
 * @var ActivationTensors::fch
 * 前馈网络隐藏层，形状为(L, B, T, 4*C)
 * @var ActivationTensors::fch_gelu
 * GELU激活后的前馈网络，形状为(L, B, T, 4*C)
 * @var ActivationTensors::fcproj
 * 前馈网络投影输出，形状为(L, B, T, C)
 * @var ActivationTensors::residual3
 * 第三个残差连接，形状为(L, B, T, C)
 * @var ActivationTensors::lnf
 * 最终层归一化输出，形状为(B, T, C)
 * @var ActivationTensors::lnf_mean
 * 最终层归一化均值，形状为(B, T)
 * @var ActivationTensors::lnf_rstd
 * 最终层归一化标准差倒数，形状为(B, T)
 * @var ActivationTensors::logits
 * 逻辑值输出，形状为(B, T, V)
 * @var ActivationTensors::probs
 * 概率输出，形状为(B, T, V)
 * @var ActivationTensors::losses
 * 损失值，形状为(B, T)
 */
typedef struct {
    float* encoded;      // (B, T, C)
    float* ln1;          // (L, B, T, C)
    float* ln1_mean;     // (L, B, T)
    float* ln1_rstd;     // (L, B, T)
    float* qkv;          // (L, B, T, 3*C)
    float* atty;         // (L, B, T, C)
    float* preatt;       // (L, B, NH, T, T)
    float* att;          // (L, B, NH, T, T)
    float* attproj;      // (L, B, T, C)
    float* residual2;    // (L, B, T, C)
    float* ln2;          // (L, B, T, C)
    float* ln2_mean;     // (L, B, T)
    float* ln2_rstd;     // (L, B, T)
    float* fch;          // (L, B, T, 4*C)
    float* fch_gelu;     // (L, B, T, 4*C)
    float* fcproj;       // (L, B, T, C)
    float* residual3;    // (L, B, T, C)
    float* lnf;          // (B, T, C)
    float* lnf_mean;     // (B, T)
    float* lnf_rstd;     // (B, T)
    float* logits;       // (B, T, V)
    float* probs;        // (B, T, V)
    float* losses;       // (B, T)
} ActivationTensors;

/**
 * @brief 分配并初始化激活张量内存
 * @param acts 激活张量结构体指针
 * @param act_sizes 各激活张量大小数组
 * @return 返回分配的激活内存块
 */
float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)malloc(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

/**
 * @struct GPT2Config
 * @brief GPT-2模型配置结构体
 * @var GPT2Config::max_seq_len
 * 最大序列长度
 * @var GPT2Config::vocab_size
 * 词汇表大小
 * @var GPT2Config::num_layers
 * 层数
 * @var GPT2Config::num_heads
 * 注意力头数
 * @var GPT2Config::channels
 * 通道数（隐藏层维度）
 */
typedef struct {
    int max_seq_len;
    int vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

/**
 * @struct GPT2
 * @brief GPT-2模型结构体
 * @var GPT2::config
 * 模型配置
 * @var GPT2::params
 * 模型参数
 * @var GPT2::param_sizes
 * 参数张量大小数组
 * @var GPT2::params_memory
 * 参数内存块
 * @var GPT2::num_parameters
 * 参数总数
 * @var GPT2::grads
 * 梯度参数
 * @var GPT2::grads_memory
 * 梯度内存块
 * @var GPT2::m_memory
 * Adam优化器m值内存
 * @var GPT2::v_memory
 * Adam优化器v值内存
 * @var GPT2::acts
 * 激活张量
 * @var GPT2::act_sizes
 * 激活张量大小数组
 * @var GPT2::acts_memory
 * 激活内存块
 * @var GPT2::num_activations
 * 激活值总数
 * @var GPT2::grads_acts
 * 激活梯度
 * @var GPT2::grads_acts_memory
 * 激活梯度内存块
 * @var GPT2::batch_size
 * 当前批次大小
 * @var GPT2::seq_len
 * 当前序列长度
 * @var GPT2::inputs
 * 输入token数组
 * @var GPT2::targets
 * 目标token数组
 * @var GPT2::mean_loss
 * 平均损失
 */
typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    int num_parameters;
    ParameterTensors grads;
    float* grads_memory;
    float* m_memory;
    float* v_memory;
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    int num_activations;
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    int batch_size;
    int seq_len;
    int* inputs;
    int* targets;
    float mean_loss;
} GPT2;

/**
 * @brief 从检查点文件构建GPT-2模型
 * @param model GPT-2模型结构体指针
 * @param checkpoint_path 检查点文件路径
 * @return 无返回值，但会填充模型参数
 */
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error: could not open model checkpoint file %s\n", checkpoint_path);
        exit(1);
    }

    int model_header[256];
    if (fread(model_header, sizeof(int), 256, model_file) != 256) {
        printf("Error: could not read model header from %s\n", checkpoint_path);
        fclose(model_file);
        exit(1);
    }

    if (model_header[0] != 20240326) {
        printf("Error: incorrect model header magic\n");
        fclose(model_file);
        exit(1);
    }
    if (model_header[1] != 1) {
        printf("Error: unsupported model version: %d\n", model_header[1]);
        fclose(model_file);
        exit(1);
    }

    int maxT, V, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];

    model->param_sizes[0] = V * C;
    model->param_sizes[1] = maxT * C;
    model->param_sizes[2] = L * C;
    model->param_sizes[3] = L * C;
    model->param_sizes[4] = L * (3 * C) * C;
    model->param_sizes[5] = L * (3 * C);
    model->param_sizes[6] = L * C * C;
    model->param_sizes[7] = L * C;
    model->param_sizes[8] = L * C;
    model->param_sizes[9] = L * C;
    model->param_sizes[10] = L * (4 * C) * C;
    model->param_sizes[11] = L * (4 * C);
    model->param_sizes[12] = L * C * (4 * C);
    model->param_sizes[13] = L * C;
    model->param_sizes[14] = C;
    model->param_sizes[15] = C;

    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    if (fread(model->params_memory, sizeof(float), num_parameters, model_file) != num_parameters) {
        printf("Error: could not read all model parameters from file\n");
        fclose(model_file);
        exit(1);
    }

    fclose(model_file);

    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

/**
 * @brief GPT-2前向传播
 * @param model GPT-2模型结构体指针
 * @param inputs 输入token IDs数组
 * @param B 批次大小
 * @param T 序列长度
 * @return 无返回值，但会更新模型激活张量
 */
void gpt2_forward(GPT2 *model, int* inputs, int B, int T) {
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    model->batch_size = B;
    model->seq_len = T;

    model->act_sizes[0] = B * T * C;
    model->act_sizes[1] = L * B * T * C;
    model->act_sizes[2] = L * B * T;
    model->act_sizes[3] = L * B * T;
    model->act_sizes[4] = L * B * T * 3*C;
    model->act_sizes[5] = L * B * T * C;
    model->act_sizes[6] = L * B * NH * T * T;
    model->act_sizes[7] = L * B * NH * T * T;
    model->act_sizes[8] = L * B * T * C;
    model->act_sizes[9] = L * B * T * C;
    model->act_sizes[10] = L * B * T * C;
    model->act_sizes[11] = L * B * T;
    model->act_sizes[12] = L * B * T;
    model->act_sizes[13] = L * B * T * 4*C;
    model->act_sizes[14] = L * B * T * 4*C;
    model->act_sizes[15] = L * B * T * C;
    model->act_sizes[16] = L * B * T * C;
    model->act_sizes[17] = B * T * C;
    model->act_sizes[18] = B * T;
    model->act_sizes[19] = B * T;
    model->act_sizes[20] = B * T * V;
    model->act_sizes[21] = B * T * V;
    model->act_sizes[22] = B * T;
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += model->act_sizes[i];
    }
    model->num_activations = num_activations;

    if (model->acts_memory) {
        free(model->acts_memory);
        model->acts_memory = NULL;
    }
    model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

    if (model->inputs) {
        free(model->inputs);
    }
    model->inputs = (int*)malloc(B * T * sizeof(int));
    memcpy(model->inputs, inputs, B * T * sizeof(int));

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

    for (int l = 0; l < L; l++) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    softmax_forward(acts.probs, acts.logits, B, T, V);
}

/**
 * @brief 将GPT-2模型梯度清零
 * @param model GPT-2模型结构体指针
 * @return 无返回值
 */
void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
    if(model->grads_acts_memory != NULL) memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
}

/**
 * @brief 释放GPT-2模型占用的内存
 * @param model GPT-2模型结构体指针
 * @return 无返回值
 */
void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

// ----------------------------------------------------------------------------
// Sampling helper (with randomness)

/**
 * @brief 根据概率分布采样
 * @param probabilities 概率数组
 * @param n 概率数组长度
 * @return 返回采样的索引
 */
int sample_mult(float* probabilities, int n) {
    // Generate a random float in [0,1)
    float coin = 0.5;  // 固定值
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // fallback
}

#define GPT2_EOT 50256

// ----------------------------------------------------------------------------
// Main interactive chat program

int main(int argc, char *argv[]) {
    TEEC_Result res;
	TEEC_Context ctx;
	TEEC_Session sess;
	TEEC_Operation op;
	TEEC_UUID uuid = TA_GPT_UUID;
	uint32_t err_origin;

	/* Initialize a context connecting us to the TEE */
	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

	/*
	 * Open a session to the "hello world" TA, the TA will print "hello
	 * world!" in the log when the session is created.
	 */
	res = TEEC_OpenSession(&ctx, &sess, &uuid,
			       TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);

	/*
	 * Execute a function in the TA by invoking it, in this case
	 * we're incrementing a number.
	 *
	 * The value of command ID part and how the parameters are
	 * interpreted is part of the interface provided by the TA.
	 */

	/* Clear the TEEC_Operation struct */
	memset(&op, 0, sizeof(op));

	/*
	 * Prepare the argument. Pass a value in the first parameter,
	 * the remaining three parameters are unused.
	 */
	op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INOUT, TEEC_NONE,
					 TEEC_NONE, TEEC_NONE);
	op.params[0].value.a = 42;

	/*
	 * TA_HELLO_WORLD_CMD_INC_VALUE is the actual function in the TA to be
	 * called.
	 */
	printf("Invoking TA to increment %d\n", op.params[0].value.a);
	res = TEEC_InvokeCommand(&sess, TA_GPT_CMD_INC_VALUE, &op,
				 &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
			res, err_origin);
	printf("TA incremented value to %d\n", op.params[0].value.a);

	/*
	 * We're done with the TA, close the session and
	 * destroy the context.
	 *
	 * The TA will print "Goodbye!" in the log when the
	 * session is closed.
	 */

	TEEC_CloseSession(&sess);

	TEEC_FinalizeContext(&ctx);


    // 默认模型路径
    const char* model_path = "models/gpt2_124M.bin";
    const char* tokenizer_path = "models/gpt2_ranks.bin";
    
    // 解析命令行参数
    if (argc == 3) {
        model_path = argv[1];
        tokenizer_path = argv[2];
    } else {
        printf("Usage: %s [model_path] [tokenizer_path]\n", argv[0]);
        return 1;
    }
    
    // 初始化随机种子
    srand((unsigned int)time(NULL));

    // 加载GPT2模型
    GPT2 model;
    gpt2_build_from_checkpoint(&model, model_path);
    printf("Model loaded from: %s\n", model_path);

    // 加载BPE ranks
    load_ranks(tokenizer_path);
    printf("Tokenizer loaded from: %s\n", tokenizer_path);

    // 读取用户提示词
    char text[1024];
    printf("Text to complete: ");
    if (!fgets(text, sizeof(text), stdin)) {
        printf("No input.\n");
        gpt2_free(&model);
        return 0;
    }
    // 去除换行
    text[strcspn(text, "\n")] = 0;

    // 转为token序列
    int tokens[MAX_TOKENS];
    int num_input_tokens = bpe_encode(text, tokens);
    if (num_input_tokens <= 0) {
        printf("Encoding failed.\n");
        gpt2_free(&model);
        return 1;
    }

    printf("\nGenerated: ");
    fflush(stdout);

    // 循环
    int current_len = num_input_tokens;
    int max_gen = 200;                         // 最大生成token数
    int max_total = model.config.max_seq_len;   // hard limit from positional embeddings

    while (current_len < max_total && current_len - num_input_tokens < max_gen) {
        // Run forward pass on current sequence
        gpt2_forward(&model, tokens, 1, current_len);

        // Get probabilities for the last position
        float* probs = model.acts.probs + (current_len - 1) * model.config.vocab_size;

        // Sample next token
        int next_token = sample_mult(probs, model.config.vocab_size);

        // Append to tokens array
        tokens[current_len] = next_token;
        current_len++;

        // Decode and print the token
        // Linear search in hash_table for the rank (could be optimised with reverse map)
        int found = 0;
        for (int i = 0; i < HASH_SIZE; i++) {
            if (hash_table[i] && hash_table[i]->rank == next_token) {
                fwrite(hash_table[i]->data, 1, hash_table[i]->len, stdout);
                fflush(stdout);
                found = 1;
                break;
            }
        }
        if (!found) {
            // If token not found in vocabulary (should not happen), print its id in brackets
            printf("[%d]", next_token);
            fflush(stdout);
        }

        // Stop if end-of-text token
        if (next_token == GPT2_EOT) break;
    }
    printf("\n");

    // Cleanup
    gpt2_free(&model);
    for (int i = 0; i < HASH_SIZE; i++) {
        if (hash_table[i]) {
            free(hash_table[i]->data);
            free(hash_table[i]);
        }
    }

    return 0;
}
