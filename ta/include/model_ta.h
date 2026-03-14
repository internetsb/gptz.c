#ifndef MODEL_TA_H
#define MODEL_TA_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "math_ta.h"

/* GPT-2模型参数结构体 */
typedef struct {
    float* wte;        // token嵌入权重，形状为(V, C)，如果trusted_layer == 1，则在TEE中计算并存储编码器输出
    float* wpe;        // 位置嵌入权重，形状为(maxT, C)，如果trusted_layer == 1，则在TEE中计算并存储编码器输出
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
} ParameterTensors_TA;

/* GPT-2模型激活张量结构体 */
typedef struct {
    float* encoded;      // encoder输出(B, T, C)，如果trusted_layer == 1，则在TEE中计算并存储编码器输出
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
    float* logits;       // 逻辑值输出(B, T, V)，如果trusted_layer == 2，则在TEE中计算并存储logits输出
    float* probs;        // 概率输出(B, T, V)，如果trusted_layer == 2，则在TEE中计算并存储概率输出
    float* losses;       // 损失值(B, T)
} ActivationTensors_TA;

#endif /* MODEL_TA_H */