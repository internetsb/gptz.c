// Original Author: Andrej Karpathy
// https://github.com/karpathy/llm.c

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#include <pthread.h>

// ----------------------------------------------------------------------------
// 所有单独层的前向传递
// B = batch_size（批次大小）, T = sequence_length（序列长度，token数）, C = channels（通道）, V = vocab_size（token词典大小）

/**
 * @brief 编码器前向传播函数
 * 
 * @param out 输出张量，形状为 (B,T,C)，在每个位置 (b,t) 存储一个 C 维向量，汇总 token 和位置信息
 * @param inp 输入张量，形状为 (B,T) 的整数数组，存储每个 (b,t) 位置的 token ID
 * @param wte Token 嵌入权重，形状为 (V,C)，V 为词汇表大小，C 为通道数
 * @param wpe 位置嵌入权重，形状为 (maxT,C)，maxT 为最大序列长度
 * @param B 批次大小
 * @param T 序列长度
 * @param C 通道数
 */
void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    // out 是 (B,T,C)。在每个位置 (b,t)，一个 C 维向量总结 token 和位置信息
    // inp 是 (B,T) 的整数，保存每个 (b,t) 位置的 token ID
    // wte 是 (V,C) 的 token 嵌入，"weight token embeddings" 的缩写
    // wpe 是 (maxT,C) 的位置嵌入，"weight positional embedding" 的缩写
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            // 定位到 out[b,t,:] 的输出位置
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            // 获取 inp[b, t] 处的 token 索引
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            // 定位到 wte 中对应这个 token 的位置
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            // 定位到 wpe 中对应这个位置的位置
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            // 将两个向量相加并将结果存储到 out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

/**
 * @brief Layer Normalization 前向传播函数
 * 
 * @param out 输出张量，形状为 (B,T,C)，归一化后的激活值
 * @param mean 输出的均值张量，形状为 (B,T)，用于反向传播
 * @param rstd 输出的标准差倒数张量，形状为 (B,T)，用于反向传播
 * @param inp 输入张量，形状为 (B,T,C) 的激活值
 * @param weight 形状为 (C) 的缩放参数
 * @param bias 形状为 (C) 的偏移参数
 * @param B 批次大小
 * @param T 序列长度
 * @param C 通道数
 */
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    // 参考：https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // inp 和 out 都是形状为 (B,T,C) 的激活值
    // mean 和 rstd 是 (B,T) 的缓冲区，用于后续的反向传播
    // 在输入的每个位置 (b,t)，C 维激活向量会先归一化，然后进行缩放和偏移
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            // 定位到输入位置 inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            // 计算均值
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            // 计算方差（没有偏差校正）
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            // 计算 rstd（标准差的倒数）
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            // 定位到 out[b,t,:] 的输出位置
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize // 归一化
                float o = n * weight[i] + bias[i]; // scale and shift // 缩放和偏移
                out_bt[i] = o; // write // 写入
            }
            // cache the mean and rstd for the backward pass later
            // 缓存均值和标准差的倒数，用于后续的反向传播
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

typedef struct {
    float* out;
    float* inp, *weight, *bias;
    size_t start;
    size_t end;
    int B, T, C, OC;
} MatmulThreadArgs;

/**
 * @brief 矩阵乘法多线程工作函数
 * 
 * @param v 指向 MatmulThreadArgs 结构体的指针
 * @return void* 返回空指针
 */
static void* matmul_worker(void* v) {
    MatmulThreadArgs* a = (MatmulThreadArgs*)v;
    size_t BT_OC = (size_t)a->T * (size_t)a->OC; // 每batch总输出任务量
    size_t BT_C = (size_t)a->T * (size_t)a->C;   // 每batch总输入任务量
    for (size_t index = a->start; index < a->end; index++) {
        size_t b = index / BT_OC;  // batch index // 批次索引
        size_t rem1 = index % BT_OC;
        size_t t = rem1 / (size_t)a->OC;  // sequence length index // 序列长度索引
        size_t o = rem1 % (size_t)a->OC;  // output channel index // 输出通道索引
        float* out_bt = a->out + b * BT_OC + t * a->OC; // OC维
        float* inp_bt = a->inp + b * BT_C + t * a->C;   // C维
        float val = (a->bias != NULL) ? a->bias[o] : 0.0f;
        float* wrow = a->weight + o * a->C;  
        for (int i = 0; i < a->C; i++) {
            val += inp_bt[i] * wrow[i];  // w*x+b
        }
        out_bt[o] = val;
    }
    return NULL;
}

#define nthreads 4

/**
 * @brief 矩阵乘法前向传播函数（多线程版本）
 * 
 * @param out 输出张量，形状为 (B, T, OC)，其中 OC 为输出通道数
 * @param inp 输入张量，形状为 (B, T, C)，其中 C 为输入通道数
 * @param weight 权重张量，形状为 (OC, C)
 * @param bias 偏置张量，形状为 (OC)，可为空(NULL)
 * @param B 批次大小
 * @param T 序列长度
 * @param C 输入通道数
 * @param OC 输出通道数
 */
void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // 输入 (B, T, C)，权重 (OC, C)，偏置 (OC)，输出 (B, T, OC)
    // 将矩阵乘法总任务量 (B*T*OC) 平摊到多个线程中
    // printf("Matmul Forward: B=%d, T=%d, C=%d, OC=%d\n", B, T, C, OC);
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
 * @brief 注意力机制前向传播函数
 * 
 * @param out 输出张量，形状为 (B, T, C)，注意力层的输出
 * @param preatt 输出的预注意力分数张量，形状为 (B, NH, T, T)，用于反向传播
 * @param att 输出的注意力分数张量，形状为 (B, NH, T, T)，用于反向传播
 * @param inp 输入张量，形状为 (B, T, 3C)，包含 Q、K、V 向量
 * @param B 批次大小
 * @param T 序列长度
 * @param C 通道数
 * @param NH 注意力头的数量
 */
void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    // 输入是形状为 (B, T, 3C) 的 QKV 向量（查询、键、值）
    // preatt, att 是形状为 (B, NH, T, T) 的矩阵。NH = 注意力头数量，T = 序列长度
    // 保存 pre-attention 和 post-attention 的得分（用于反向传播）
    // 输出是形状为 (B, T, C) 的矩阵
    // attention 是唯一跨时间步混合信息的层
    // 所有其他操作都在每个 (b,t) 位置独立应用
    // （当然，没有层会在批次之间混合信息）
    int C3 = C*3;
    int hs = C / NH; // head size // 注意力头尺寸
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                // 第一遍：计算 query 与 key 的点积和最大值
                float maxval = -10000.0f; // TODO something better // TODO 更好的方法
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key // +C 因为这是 key

                    // (query_t) dot (key_t2)
                    // (query_t) 与 (key_t2) 的点积
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                // 第二遍：计算指数并跟踪总和
                // maxval 仅为了数值稳定性而被计算和减去
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                // 第三遍：归一化得到 softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        // 因果注意力掩码。不一定需要在此处设为零
                        // 仅为了调试和与 PyTorch 对比而显式执行此操作
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                // 第四遍：将加权值累积到注意力输出中
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value // +C*2 因为这是 value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}
# define M_PI		3.14159265358979323846	/* pi */
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

/**
 * @brief GeLU 激活函数前向传播
 * 
 * @param out 输出张量，形状为 (N)，GeLU 激活后的结果
 * @param inp 输入张量，形状为 (N)，待激活的输入
 * @param N 张量元素总数
 */
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    // Transformer 的 MLP 块中的 (近似) GeLU 逐元素非线性激活函数
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

/**
 * @brief 残差连接前向传播函数
 * 
 * @param out 输出张量，形状为 (N)，存储 inp1 和 inp2 的逐元素相加结果
 * @param inp1 输入张量1，形状为 (N)
 * @param inp2 输入张量2，形状为 (N)
 * @param N 张量元素总数
 */
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

/**
 * @brief Softmax 前向传播函数
 * 
 * @param probs 输出的概率张量，形状为 (B,T,V)，每项和为1
 * @param logits 输入的未归一化对数概率张量，形状为 (B,T,V)
 * @param B 批次大小
 * @param T 序列长度
 * @param V 词汇表大小
 */
void softmax_forward(float* probs, float* logits, int B, int T, int V) {
    // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,V) of the unnormalized log probabilities
    // 输出：probs 是形状为 (B,T,V) 的概率（在每个 b,t 位置总和为 1.0）
    // 输入：logits 是形状为 (B,T,V) 的未归一化对数概率
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            // maxval is only calculated and subtracted for numerical stability
            // maxval 仅为了数值稳定性而被计算和减去
            float maxval = -10000.0f; // TODO something better // TODO 更好的方法
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
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
// GPT-2 model definition
// GPT-2 模型定义

// the parameters of the model
// 模型的参数
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

/**
 * @brief 为模型参数分配内存并将张量指针指向正确位置
 * 
 * @param params ParameterTensors 结构体指针，存储模型参数
 * @param param_sizes 存储各参数张量大小的数组
 * @return float* 指向分配的参数内存块的指针
 */
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    // 一次性分配所有参数的内存
    float* params_memory = (float*)malloc(num_parameters * sizeof(float));
    // assign all the tensors
    // 为所有张量分配空间
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
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

/**
 * @brief 为模型激活值分配内存并将张量指针指向正确位置
 * 
 * @param acts ActivationTensors 结构体指针，存储模型激活值
 * @param act_sizes 存储各激活张量大小的数组
 * @return float* 指向分配的激活内存块的指针
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

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
    // 最大序列长度，例如 1024
    // 词汇表大小，例如 50257
    // 层数，例如 12
    // 注意力头数，例如 12
    // 通道数，例如 768
} GPT2Config;

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    // 模型的权重（参数）及其大小
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    int num_parameters;
    // gradients of the weights
    // 权重梯度
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    // AdamW 优化器的缓冲区
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    // 模型的激活值及其大小
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    int num_activations;
    // gradients of the activations
    // 激活值梯度
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    // 其他运行状态配置
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    // 当前前向传播的批次大小 (B)
    // 当前前向传播的序列长度 (T)
    // 当前前向传播的输入 token
    // 当前前向传播的目标 token
    // 前向传播完成后，将填入平均损失值
} GPT2;

/**
 * @brief 从检查点文件构建 GPT-2 模型
 * 
 * @param model GPT2 模型结构体指针
 * @param checkpoint_path 检查点文件路径
 */
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    // 从检查点文件读取模型
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error: could not open model checkpoint file %s\n", checkpoint_path);
        exit(1);
    }

    // read in header information
    // 读取头信息
    int model_header[256];
    if (fread(model_header, sizeof(int), 256, model_file) != 256) {
        printf("Error: could not read model header from %s\n", checkpoint_path);
        fclose(model_file);
        exit(1);
    }

    // validate header
    // 验证头信息
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

    // read in hyperparameters
    // 读取超参数
    int maxT, V, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];

    // allocate space for all the parameters and read them in
    // 为所有参数分配空间并读取它们
    model->param_sizes[0] = V * C; // wte
    model->param_sizes[1] = maxT * C; // wpe
    model->param_sizes[2] = L * C; // ln1w
    model->param_sizes[3] = L * C; // ln1b
    model->param_sizes[4] = L * (3 * C) * C; // qkvw
    model->param_sizes[5] = L * (3 * C); // qkvb
    model->param_sizes[6] = L * C * C; // attprojw
    model->param_sizes[7] = L * C; // attprojb
    model->param_sizes[8] = L * C; // ln2w
    model->param_sizes[9] = L * C; // ln2b
    model->param_sizes[10] = L * (4 * C) * C; // fcw
    model->param_sizes[11] = L * (4 * C); // fcb
    model->param_sizes[12] = L * C * (4 * C); // fcprojw
    model->param_sizes[13] = L * C; // fcprojb
    model->param_sizes[14] = C; // lnfw
    model->param_sizes[15] = C; // lnfb

    // count the number of parameters
    // 计算参数数量
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    // 从文件读取所有参数
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    if (fread(model->params_memory, sizeof(float), num_parameters, model_file) != num_parameters) {
        printf("Error: could not read all model parameters from file\n");
        fclose(model_file);
        exit(1);
    }

    fclose(model_file);

    // other inits
    // 其他初始化
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss // -1.0f 表示无损失值
}

/**
 * @brief GPT-2 模型前向传播
 * 
 * @param model GPT2 模型结构体指针
 * @param inputs 输入 token 数组，形状为 (B, T)
 * @param B 批次大小
 * @param T 序列长度
 */
void gpt2_forward(GPT2 *model, int* inputs, int B, int T) {
    // convenience parameters
    // 便捷参数
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // record the current B,T as well
    // 同时记录当前的 B,T
    model->batch_size = B;
    model->seq_len = T;
    // and now allocate the space
    // 然后分配空间
    model->act_sizes[0] = B * T * C; // encoded
    model->act_sizes[1] = L * B * T * C; // ln1
    model->act_sizes[2] = L * B * T;  // ln1_mean
    model->act_sizes[3] = L * B * T;  // ln1_rstd
    model->act_sizes[4] = L * B * T * 3*C; // qkv
    model->act_sizes[5] = L * B * T * C;  // atty
    model->act_sizes[6] = L * B * NH * T * T;  // preatt
    model->act_sizes[7] = L * B * NH * T * T;  // att
    model->act_sizes[8] = L * B * T * C; // attproj
    model->act_sizes[9] = L * B * T * C; // residual2
    model->act_sizes[10] = L * B * T * C; // ln2
    model->act_sizes[11] = L * B * T; // ln2_mean
    model->act_sizes[12] = L * B * T; // ln2_rstd
    model->act_sizes[13] = L * B * T * 4*C; // fch
    model->act_sizes[14] = L * B * T * 4*C; // fch_gelu
    model->act_sizes[15] = L * B * T * C; // fcproj
    model->act_sizes[16] = L * B * T * C; // residual3
    model->act_sizes[17] = B * T * C; // lnf
    model->act_sizes[18] = B * T; // lnf_mean
    model->act_sizes[19] = B * T; // lnf_rstd
    model->act_sizes[20] = B * T * V; // logits
    model->act_sizes[21] = B * T * V; // probs
    model->act_sizes[22] = B * T; // losses
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

    // also create memory for caching inputs and targets
    // 同时创建缓存输入和目标的内存
    if (model->inputs) {
        free(model->inputs);
    }
    model->inputs = (int*)malloc(B * T * sizeof(int));

    // cache the inputs/targets
    // 缓存输入/目标
    memcpy(model->inputs, inputs, B * T * sizeof(int));

    // forward pass
    // 前向传播
    ParameterTensors params = model->params; // for brevity // 为简洁起见
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    // 编码结果进入 residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        // 获取该层权重的指针
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

        // get the pointers of the activations for this layer
        // 获取该层激活值的指针
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

        // now do the forward pass
        // 现在执行前向传播
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
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    // 最后的残差连接在 residual3 中
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    softmax_forward(acts.probs, acts.logits, B, T, V);
}

/**
 * @brief 将 GPT-2 模型的梯度清零
 * 
 * @param model GPT2 模型结构体指针
 */
void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

/**
 * @brief 释放 GPT-2 模型占用的内存
 * 
 * @param model GPT2 模型结构体指针
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

/**
 * @brief 根据概率分布采样索引
 * 
 * @param probabilities 概率数组，必须和为1
 * @param n 概率数组的长度
 * @return int 采样的索引
 */
int sample_mult(float* probabilities, int n) {
    // sample index from probabilities (they must sum to 1!)
    // coin can be a random number in [0, 1), usually from random_f32()
    // 从概率分布中采样索引（概率之和必须为 1！）
    // 硬币可以是 [0, 1) 范围内的随机数，通常来自 random_f32()
    float cdf = 0.0f, coin = 0.5f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
    // 防止舍入误差导致的问题
}

// the GPT-2 end-of-text token id
// GPT-2 文本结束标记 ID
#define GPT2_EOT 50256

/**
 * @brief 主函数，GPT-2模型推理入口
 * 
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return int 程序退出状态码
 */
int main(int argc, char** argv) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "models/gpt2_124M.bin");
    const int n = 10;  // Token limit.

    if (argc == 1) {
        printf("Provide at least one token.\n");
        exit(1);
    }
    if (argc > n) {
        printf("Too many tokens.\n");
        exit(1);
    }

    int tokens[n];

    for (int i = 0; i < n; i++) {
        if (i + 1 < argc) {
            tokens[i] = strtol(argv[i + 1], NULL, 10);
        } else {
            tokens[i] = GPT2_EOT;
        }
    }

    for (int t = argc - 1; t < n; t++) {
        gpt2_forward(&model, tokens, 1, t);
        float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
        int next_token = sample_mult(probs, model.config.vocab_size);
        tokens[t] = next_token;

        printf("%d\n", tokens[t]);
        fflush(stdout);
    }

    gpt2_free(&model);

    return 0;
}