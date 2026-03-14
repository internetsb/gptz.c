#ifndef GPT_H
#define GPT_H

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
#include <err.h>
// -------------------------tokenizer 结构体---------------------------------------------------

/* 存储字节序列及其排名的结构体，用于BPE排序*/
typedef struct {
    unsigned char* data;  //字节序列数据
    int len;              //数据长度
    int rank;             //排名值
} RankItem;

/* 临时存储token的结构体，用于BPE合并 */
typedef struct {
    unsigned char* bytes; //字节序列指针
    int len;              //长度
} Token;

// --------------------------GPT-2 模型结构体--------------------------------------------------

#define NUM_PARAMETER_TENSORS 16
/* GPT-2模型参数张量结构体 */
typedef struct {
    float* wte;        // token嵌入权重，形状为(V, C)
    float* wpe;        // 位置嵌入权重，形状为(maxT, C)
    float* ln1w;       // 第一层归一化权重，形状为(L, C)
    float* ln1b;       // 第一层归一化偏置，形状为(L, C)
    float* qkvw;       // QKV变换权重，形状为(L, 3*C, C)
    float* qkvb;       // QKV变换偏置，形状为(L, 3*C)
    float* attprojw;   // 注意力投影权重，形状为(L, C, C)
    float* attprojb;   // 注意力投影偏置，形状为(L, C)
    float* ln2w;       // 第二层归一化权重，形状为(L, C)
    float* ln2b;       // 第二层归一化偏置，形状为(L, C)
    float* fcw;        // 前馈网络权重，形状为(L, 4*C, C)
    float* fcb;        // 前馈网络偏置，形状为(L, 4*C)
    float* fcprojw;    // 前馈网络投影权重，形状为(L, C, 4*C)
    float* fcprojb;    // 前馈网络投影偏置，形状为(L, C)
    float* lnfw;       // 最终层归一化权重，形状为(C)
    float* lnfb;       // 最终层归一化偏置，形状为(C)
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 23
/* GPT-2模型激活张量结构体 */
typedef struct {
    float* encoded;      // encoder输出(B, T, C)
    float* ln1;          // 第一层归一化输出(L, B, T, C)
    float* ln1_mean;     // 第一层归一化均值(L, B, T)
    float* ln1_rstd;     // 第一层归一化标准差倒数(L, B, T)
    float* qkv;          // QKV变换结果(L, B, T, 3*C)
    float* atty;         // 注意力输出(L, B, T, C)
    float* preatt;       // 注意力预激活值(L, B, NH, T, T)
    float* att;          // 注意力权重(L, B, NH, T, T)
    float* attproj;      // 注意力投影输出(L, B, T, C)
    float* residual2;    // 第二个残差连接(L, B, T, C)
    float* ln2;          // 第二层归一化输出(L, B, T, C)
    float* ln2_mean;     // 第二层归一化均值(L, B, T)
    float* ln2_rstd;     // 第二层归一化标准差倒数(L, B, T)
    float* fch;          // 前馈网络隐藏层(L, B, T, 4*C)
    float* fch_gelu;     // GELU激活后的前馈网络(L, B, T, 4*C)
    float* fcproj;       // 前馈网络投影输出(L, B, T, C)
    float* residual3;    // 第三个残差连接(L, B, T, C)
    float* lnf;          // 最终层归一化输出(B, T, C)
    float* lnf_mean;     // 最终层归一化均值(B, T)
    float* lnf_rstd;     // 最终层归一化标准差倒数(B, T)
    float* logits;       // 逻辑值输出(B, T, V)
    float* probs;        // 概率输出(B, T, V)
    float* losses;       // 损失值(B, T)
} ActivationTensors;

/* GPT-2模型配置结构体 */
typedef struct {
    int max_seq_len;    // 最大序列长度
    int vocab_size;     // 词汇表大小
    int num_layers;     // 注意力层数
    int num_heads;      // 注意力头数
    int channels;       // 通道数（隐藏层维度）
} GPT2Config;

/* GPT-2模型结构体 */
typedef struct {
    GPT2Config config;                          // 模型配置
    ParameterTensors params;                    // 模型参数
    size_t param_sizes[NUM_PARAMETER_TENSORS];  // 参数张量大小数组
    float* params_memory;                       // 参数内存块
    int num_parameters;                         // 参数总数
    ParameterTensors grads;                     // 梯度参数
    float* grads_memory;                        // 梯度内存块
    float* m_memory;                            // Adam优化器m值内存
    float* v_memory;                            // Adam优化器v值内存
    ActivationTensors acts;                     // 激活张量
    size_t act_sizes[NUM_ACTIVATION_TENSORS];   // 激活张量大小数组
    float* acts_memory;                         // 激活值内存块，存储前向传播过程中的激活值
    int num_activations;                        // 激活值总数
    ActivationTensors grads_acts;               // 激活值的梯度内存块
    float* grads_acts_memory;                   // 激活值的梯度内存块
    int batch_size;                             // 批处理大小，当前批次中的样本数量
    int seq_len;                                // 序列长度，当前处理序列的实际长度
    int* inputs;                                // 输入数据数组，存储输入序列的token索引
    int* targets;                               // 目标数据数组，存储期望的输出标签
    float mean_loss;                            // 平均损失值，初始化为-1.0f表示尚未计算
    int trusted_layer;                          // 存入TA的层，1表示encoder层，2表示softmax层
} GPT2;

#endif // GPT_H