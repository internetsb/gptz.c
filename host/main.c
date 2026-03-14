// Original Author: Andrej Karpathy
// https://github.com/karpathy/llm.c
//
// 所需文件:
//   gpt2_124M.bin   - GPT-2 模型文件
//   gpt2_ranks.bin   - BPE ranks 文件 (由 export_tokenizer.py 导出)

#include <gpt.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* 获取UUID */
#include <gpt_ta.h>

TEEC_Session sess;
TEEC_Context ctx;
// ----------------------------------------------------------------------------
// BPE tokenizer 结构和函数
#define HASH_SIZE 131072   // 2^17, 必须比词表大小大
#define MAX_TOKENS 2048    // 我们处理token的最大长度

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

        // 合并
        int new_len = tokens[best_idx].len + tokens[best_idx+1].len;
        unsigned char* new_data = malloc(new_len);
        memcpy(new_data, tokens[best_idx].bytes, tokens[best_idx].len);
        memcpy(new_data + tokens[best_idx].len, tokens[best_idx+1].bytes, tokens[best_idx+1].len);
        
        free(tokens[best_idx].bytes);
        free(tokens[best_idx+1].bytes);
        tokens[best_idx].bytes = new_data;
        tokens[best_idx].len = new_len;

        // 左移其余token
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
// GPT-2 模型定义及前向传播

void encoder_forward_CA(int* inputs, int B, int T, int C){
    int inputs_size = B * T * sizeof(int);
    int* inputs_buffer = malloc(inputs_size);
    memcpy(inputs_buffer, inputs, inputs_size);

    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_VALUE_INPUT, TEEC_NONE);
    op.params[0].tmpref.buffer = inputs_buffer;
    op.params[0].tmpref.size = (uint32_t)inputs_size;
    op.params[1].value.a = (uint32_t)B;
    op.params[1].value.b = (uint32_t)T;
    op.params[2].value.a = (uint32_t)C;

    res = TEEC_InvokeCommand(&sess, TA_GPT_CMD_ENCODER_FORWARD, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        errx(1, "TEE_InvokeCommand failed with code 0x%x origin 0x%x", res, err_origin);
    }

    free(inputs_buffer);
}

void encoder_output_CA(float* out, int B, int T, int C) {
    int out_size = B * T * C * sizeof(float);
    float* out_buffer = malloc(out_size);
    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
    op.params[0].tmpref.buffer = out_buffer;
    op.params[0].tmpref.size = (uint32_t)out_size;

    res = TEEC_InvokeCommand(&sess, TA_GPT_CMD_ENCODER_OUTPUT, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        errx(1, "TEE_InvokeCommand failed with code 0x%x origin 0x%x", res, err_origin);
    }
    memcpy(out, out_buffer, out_size);

    free(out_buffer);
}

/**
 * @brief 编码器前向传播：token嵌入 + 位置嵌入
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

// matmul_forward 的线程参数结构
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

                // 1: 计算query点乘key，和maxval
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

                // 2： 幂和
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // 3: softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                // 4: values 的加权和
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

// 在TA实现
// matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
// softmax_forward(acts.probs, acts.logits, B, T, V);
void matmul_softmax_forward_CA(float* inputs, int B, int T, int C, int V){
    int inputs_size = B * T * C * sizeof(float);
    float* inputs_buffer = malloc(inputs_size);
    memcpy(inputs_buffer, inputs, inputs_size);

    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_VALUE_INPUT, TEEC_NONE);
    op.params[0].tmpref.buffer = inputs;
    op.params[0].tmpref.size = (uint32_t)inputs_size;
    op.params[1].value.a = (uint32_t)B;
    op.params[1].value.b = (uint32_t)T;
    op.params[2].value.a = (uint32_t)C;
    op.params[2].value.b = (uint32_t)V;

    res = TEEC_InvokeCommand(&sess, TA_GPT_CMD_SOFTMAX_FORWARD, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x\n", res, err_origin);
    }
    free(inputs_buffer);
}


void softmax_output_CA(float *output, int B, int T, int V){
    int outputs_size = B * T * V * sizeof(float);
    float *outputs_buffer = malloc(outputs_size);

    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
    op.params[0].tmpref.buffer = outputs_buffer;
    op.params[0].tmpref.size = (uint32_t)outputs_size;

    res = TEEC_InvokeCommand(&sess, TA_GPT_CMD_SOFTMAX_OUTPUT, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x\n", res, err_origin);
    }
    memcpy(output, outputs_buffer, outputs_size);
    free(outputs_buffer);
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
// GPT-2 内存分配

/**
 * @brief 分配并初始化参数张量内存
 * @param params 参数张量结构体指针
 * @param param_sizes 各参数张量大小数组
 * @param trusted_layer 可信标志，0-全部在普通环境中运行，1-编码器部分softmax部分在TEE中运行
 * @return 返回分配的参数内存块
 */
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, int trusted_layer) {
    size_t num_parameters = 0;
        for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    float* params_memory = (float*)malloc(num_parameters * sizeof(float)); // 参数内存块
    if (!params_memory) {
        fprintf(stderr, "Error: could not allocate memory for parameters\n");
        exit(1);
    }
    
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        if (trusted_layer == 1 && (i == 0 || i == 1)) { // 如果可信标志为1，且参数是wte或wpe
            *(ptrs[i]) = NULL; // 将wte和wpe的指针设置为NULL，表示它们将在TEE中加载和计算
            params_memory_iterator += param_sizes[i];
            continue; // 下一个参数继续分配内存
        }
        *(ptrs[i]) = params_memory_iterator;    // 将相应参数内存指针指向当前位置
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}



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

void load_parameters_CA(GPT2 *model, FILE *model_file) { 
    // 读入wte和wpe参数数据到缓冲区，准备发送到TEE中
    int buffer_size = model->param_sizes[0] + model->param_sizes[1]; // wte和wpe的总大小
    float* buffer = (float*)malloc(buffer_size * sizeof(float));
    if (fread(buffer, sizeof(float), buffer_size, model_file) != (size_t)buffer_size) {
        printf("Error: could not read wte and wpe parameters from file\n");
        fclose(model_file);
        free(buffer);
        exit(1);
    }
    printf("Ready to load parameters into TEE...\n");
    // TA函数调用
    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;
    // 参数：模型参数内存块指针，wte和wpe大小
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_NONE, TEEC_NONE);
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = (int)(buffer_size * sizeof(float));
    op.params[1].value.a = (int)model->param_sizes[0]; // wte大小
    op.params[1].value.b = (int)model->param_sizes[1]; // wpe大小

    res = TEEC_InvokeCommand(&sess, TA_GPT_CMD_LOAD_PARAMS, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        errx(1, "TEE_InvokeCommand failed with code 0x%x origin 0x%x", res, err_origin);
    }
}


/** 
 * @brief 从检查点文件加载模型参数
 * @param model GPT-2模型结构体指针
 * @param model_file 已打开的模型检查点文件指针
 * @param trusted_layer 可信标志，0-全部加载入普通环境，1-编码器部分softmax部分参数wte,wpe在TEE中加载
 * @return 无返回值，但会填充模型参数内存空间
 */
void load_parameters(GPT2 *model, FILE *model_file, int trusted_layer) {
    // 依次加载参数数据
    float* params_memory_iterator = model->params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        if (trusted_layer == 1 && (i == 0)) {
            // 如果当前参数是wte或wpe，并且可信标志为1，加载入TEE中
            load_parameters_CA(model, model_file); // 连续加载wte和wpe，将参数加载到TEE中
            printf("Loaded parameters into TEE...\n");
            params_memory_iterator += model->param_sizes[0] + model->param_sizes[1]; // 跳过这部分内存
            i = 1; // 跳过wpe的索引，因为它已经在load_parameters_CA中加载了
            continue; // 下一个参数继续加载
        }
        if (fread(params_memory_iterator, sizeof(float), model->param_sizes[i], model_file) != model->param_sizes[i]) {
            printf("Error: could not read parameter %zu from file\n", i);
            fclose(model_file);
            exit(1);
        }
        printf("Loaded parameter %zu...\n", i);
        params_memory_iterator += model->param_sizes[i];
    }

}

/**
 * @brief 从检查点文件构建GPT-2模型
 * @param model GPT-2模型结构体指针
 * @param checkpoint_path 检查点文件路径
 * @param trusted_layer 可信标志，0-全部加载入普通环境，1-编码器部分softmax部分参数wte,wpe在TEE中加载
 * @return 无返回值，但会填充模型参数
 */
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, int trusted_layer) {
    // 读取检查点文件
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error: could not open model checkpoint file %s\n", checkpoint_path);
        exit(1);
    }
    // 校验头部
    int model_header[256];
    if (fread(model_header, sizeof(int), 256, model_file) != 256) {  // 读取模型头
        printf("Error: could not read model header from %s\n", checkpoint_path);
        fclose(model_file);
        exit(1);
    }
    if (model_header[0] != 20240326) {  // 模型头校验魔数
        printf("Error: incorrect model header magic\n");
        fclose(model_file);
        exit(1);
    }
    if (model_header[1] != 1) {  // 模型头校验版本
        printf("Error: unsupported model version: %d\n", model_header[1]);
        fclose(model_file);
        exit(1);
    }

    int maxT, V, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2]; // 最大序列长度
    model->config.vocab_size = V = model_header[3];  // 词表大小
    model->config.num_layers = L = model_header[4];  // 注意力层数
    model->config.num_heads = NH = model_header[5];  // 注意力头数
    model->config.channels = C = model_header[6];  // 通道数（隐藏层维度）
    // 所有待加载的参数大小
    model->param_sizes[0] = V * C;          // token嵌入权重 (wte): 将词汇索引转换为向量表示
    model->param_sizes[1] = maxT * C;       // 位置嵌入权重 (wpe): 为序列中的每个位置编码位置信息
    model->param_sizes[2] = L * C;          // 第一层归一化权重 (ln1w): 每层第一个LayerNorm的缩放参数
    model->param_sizes[3] = L * C;          // 第一层归一化偏置 (ln1b): 每层第一个LayerNorm的偏移参数
    model->param_sizes[4] = L * (3 * C) * C;// QKV变换权重 (qkvw): 计算查询、键、值的权重矩阵 (3代表Q,K,V三个矩阵)
    model->param_sizes[5] = L * (3 * C);    // QKV变换偏置 (qkvb): 计算查询、键、值的偏置向量
    model->param_sizes[6] = L * C * C;      // 注意力投影权重 (attprojw): 多头注意力后的线性投影权重
    model->param_sizes[7] = L * C;          // 注意力投影偏置 (attprojb): 多头注意力后的线性投影偏置
    model->param_sizes[8] = L * C;          // 第二层归一化权重 (ln2w): 每层第二个LayerNorm的缩放参数
    model->param_sizes[9] = L * C;          // 第二层归一化偏置 (ln2b): 每层第二个LayerNorm的偏移参数
    model->param_sizes[10] = L * (4 * C) * C;// 前馈网络权重 (fcw): Transformer前馈网络的第一层权重 (4C是中间层维度)
    model->param_sizes[11] = L * (4 * C);   // 前馈网络偏置 (fcb): Transformer前馈网络的第一层偏置
    model->param_sizes[12] = L * C * (4 * C);// 前馈网络投影权重 (fcprojw): 前馈网络降维投影的权重
    model->param_sizes[13] = L * C;         // 前馈网络投影偏置 (fcprojb): 前馈网络降维投影的偏置
    model->param_sizes[14] = C;             // 最终层归一化权重 (lnfw): 最后输出前LayerNorm的缩放参数
    model->param_sizes[15] = C;             // 最终层归一化偏置 (lnfb): 最后输出前LayerNorm的偏移参数
    // 总参数大小
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;
    // 参数内存分配
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, trusted_layer);
    printf ("Allocated %zu bytes for model parameters\n", num_parameters * sizeof(float));
    load_parameters(model, model_file, trusted_layer); // 从文件加载参数到内存，考虑trusted_layer的情况
    printf ("Loaded %zu bytes of model parameters\n", num_parameters * sizeof(float));

    fclose(model_file);

    model->acts_memory = NULL;          // 激活内存
    model->grads_memory = NULL;         // 梯度内存
    model->m_memory = NULL;             // Adam优化器m内存
    model->v_memory = NULL;             // Adam优化器v内存
    model->grads_acts_memory = NULL;    // 激活值梯度内存
    model->inputs = NULL;               // 输入序列的token索引
    model->targets = NULL;              // 存储期望的输出标签
    model->batch_size = 0;              // 当前批次的数据量
    model->seq_len = 0;                 // 当前输入序列长度
    model->mean_loss = -1.0f;           // 平均损失值，初始化为-1.0f表示尚未计算
    model->trusted_layer = 0;          // 存入TA的层
}

/**
 * @brief GPT-2前向传播
 * @param model GPT-2模型结构体指针
 * @param inputs 输入token IDs数组
 * @param B 批次大小
 * @param T 序列长度
 * @param trusted_layer 可信标志，0-全部在普通环境中运行，1-编码器部分softmax部分在TEE中运行
 * @return 无返回值，但会更新模型激活张量
 */
void gpt2_forward(GPT2 *model, int* inputs, int B, int T, int trusted_layer) {
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    model->batch_size = B;
    model->seq_len = T;

    model->act_sizes[0] = B * T * C;  // encoded
    model->act_sizes[1] = L * B * T * C;  // ln1
    model->act_sizes[2] = L * B * T;  // ln1_mean
    model->act_sizes[3] = L * B * T;  // ln1_rstd
    model->act_sizes[4] = L * B * T * 3*C;  // qkv
    model->act_sizes[5] = L * B * T * C;  // atty
    model->act_sizes[6] = L * B * NH * T * T;  // preatt
    model->act_sizes[7] = L * B * NH * T * T;  // att
    model->act_sizes[8] = L * B * T * C;  // attproj
    model->act_sizes[9] = L * B * T * C;  // residual2
    model->act_sizes[10] = L * B * T * C;  // ln2
    model->act_sizes[11] = L * B * T;  // ln2_mean
    model->act_sizes[12] = L * B * T;  // ln2_rstd
    model->act_sizes[13] = L * B * T * 4*C;  // fch
    model->act_sizes[14] = L * B * T * 4*C;  // fch_gelu
    model->act_sizes[15] = L * B * T * C;  // fcproj
    model->act_sizes[16] = L * B * T * C;  // residual3
    model->act_sizes[17] = B * T * C;  // lnf
    model->act_sizes[18] = B * T;  // lnf_mean
    model->act_sizes[19] = B * T;  // lnf_rstd
    model->act_sizes[20] = B * T * V;  // logits
    model->act_sizes[21] = B * T * V;  // probs
    model->act_sizes[22] = B * T;  // losses

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
    
    // 如果trusted_layer == 1，则编码器部分在TEE中完成
    if (trusted_layer == 1) {
        encoder_forward_CA(inputs, B, T, C); // 在TEE中完成
        encoder_output_CA(acts.encoded, B, T, C); // 获取encoder输出存入acts.encoded
    } else {
        encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
    }
    // 注意力层前馈网络
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
    
    // 如果trusted_layer == 1，则最终的matmul和softmax在TEE中完成
    if (trusted_layer == 1) {
        // 通过TEE接口完成最后的计算      
        // matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
        // softmax_forward(acts.probs, acts.logits, B, T, V);
        matmul_softmax_forward_CA(acts.lnf, B, T, C, V); // 在TEE中完成matmul和softmax
        softmax_output_CA(acts.probs, B, T, V); // 获取softmax输出存入acts.probs
    } else {
        matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
        softmax_forward(acts.probs, acts.logits, B, T, V);
    }
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
// 随机采样函数

/**
 * @brief 根据概率分布采样
 * @param probabilities 概率数组
 * @param n 概率数组长度
 * @return 返回采样的索引
 */
int sample_mult(float* probabilities, int n) {
    float coin = 0.5;  // 固定值，可修改为随机值
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
// 主交互程序
void prepare_tee_session()
{
    TEEC_UUID uuid = TA_GPT_UUID;
    uint32_t err_origin;
    TEEC_Result res;

    /* 初始化上下文 */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

    /* 打开会话 */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
         res, err_origin);
}

void terminate_tee_session()
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}

int main(int argc, char *argv[]) {
    prepare_tee_session();
    printf("Session started.\n");
    // 默认模型路径
    const char* model_path = "models/gpt2_124M.bin";
    const char* tokenizer_path = "models/gpt2_ranks.bin";
    int trusted_layer_flag = 0;
    
    // 解析命令行参数
    if (argc == 3) {
        model_path = argv[1];
        tokenizer_path = argv[2];
    } else if (argc == 4 && strcmp(argv[3], "-T") == 0) {
        model_path = argv[1];
        tokenizer_path = argv[2];
        trusted_layer_flag = 1;
        printf("Trusted layer enabled.\n");
    } else {
        printf("Usage: %s <model_path> <tokenizer_path> [-T]\n", argv[0]);
        return 1;
    }
    // 初始化随机种子
    srand((unsigned int)time(NULL));

    // 加载GPT2模型
    GPT2 model;
    gpt2_build_from_checkpoint(&model, model_path, trusted_layer_flag);
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
    int max_total = model.config.max_seq_len;   // 位置嵌入的极限

    while (current_len < max_total && current_len - num_input_tokens < max_gen) {
        // 以当前token序列前向传播
        gpt2_forward(&model, tokens, 1, current_len, trusted_layer_flag);

        // 获取概率
        float* probs = model.acts.probs + (current_len - 1) * model.config.vocab_size;

        // 采样
        int next_token = sample_mult(probs, model.config.vocab_size);

        // 添加到token数组
        tokens[current_len] = next_token;
        current_len++;

        // 解码并输出
        // 线性查找（可优化）
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
            // 如果token在词表中找不到 (不应发生), 打印id
            printf("[%d]", next_token);
            fflush(stdout);
        }

        // 遇到结束token停止
        if (next_token == GPT2_EOT) break;
    }
    printf("\n");

    // 清理资源
    terminate_tee_session();
    gpt2_free(&model);
    for (int i = 0; i < HASH_SIZE; i++) {
        if (hash_table[i]) {
            free(hash_table[i]->data);
            free(hash_table[i]);
        }
    }

    return 0;
}
