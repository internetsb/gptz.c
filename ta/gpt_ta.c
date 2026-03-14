// SPDX-License-Identifier: BSD-2-Clause
/*
 * Copyright (c) 2016, Linaro Limited
 * All rights reserved.
 */

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <gpt_ta.h>
#include <model_ta.h>

ParameterTensors_TA param_tensors;
ActivationTensors_TA act_tensors;
// ------------------------ TA 基本函数--------------------------

/* 当TA实例被创建时调用。这是TA的第一个调用 */
TEE_Result TA_CreateEntryPoint(void)
{
	DMSG("has been called");

	return TEE_SUCCESS;
}

/* 当TA实例被销毁时调用。这是TA的最后一个调用，前提是TA没有崩溃。 */
void TA_DestroyEntryPoint(void)
{
	DMSG("has been called");
}

/* 
 * 当一个新的会话被打开到TA时调用。
 * sess_ctx可以更新为能够识别此会话的会话上下文。在这个函数中，您通常会进行TA的全局初始化。 
 */
TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
				    TEE_Param __unused params[4],
				    void __unused **sess_ctx)
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	IMSG("Initialized\n");

	/* 如果返回值不等于TEE_SUCCESS，则不会创建会话。*/
	return TEE_SUCCESS;
}

/* 当一个会话被关闭时调用。sess_ctx 值是 TA_OpenSessionEntryPoint() 中设置的 */
void TA_CloseSessionEntryPoint(void __unused *sess_ctx)
{
	if (param_tensors.wte) {
		free(param_tensors.wte);
	}
	if (param_tensors.wpe) {
		free(param_tensors.wpe);
	}
	if (act_tensors.encoded) {
		free(act_tensors.encoded);
	}
	if (act_tensors.logits) {
		free(act_tensors.logits);
	} 
	if (act_tensors.probs) {
		free(act_tensors.probs);
	}
	IMSG("Resources freed\n");
	IMSG("Goodbye!\n");
}
// ------------------------ TA 函数实现 --------------------------

/**
 * @brief 矩阵乘法前向传播（单线程）
 * @param[out] out 输出张量，形状为(B, T, OC)
 * @param inp 输入张量，形状为(B, T, C)
 * @param weight 权重矩阵，形状为(C, OC)
 * @param bias 偏置向量，形状为(OC)，可为NULL
 * @param B 批次大小
 * @param T 序列长度
 * @param C 输入通道数
 * @param OC 输出通道数
 */
static void matmul_forward_TA(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o * C;
                
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                
                out_bt[o] = val;
            }
        }
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
static void softmax_forward_TA(float* probs, float* logits, int B, int T, int V) {
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
                probs_bt[i] = (float)ta_exp((double)(logits_bt[i] - maxval));
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}
// -------------------- TA 命令入口 --------------------------
static TEE_Result load_parameters_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	// 解析参数
	float* buffer = (float*)params[0].memref.buffer;
	size_t buffer_size = params[0].memref.size;
	int wte_size = params[1].value.a;
	// 分配内存
    float *params_memory = malloc(buffer_size);
	if (!params_memory) {
		EMSG("Out of memory");
		return TEE_ERROR_OUT_OF_MEMORY;
	}
	memcpy(params_memory, buffer, buffer_size);

	// 将参数指针指向相应位置
	param_tensors.wte = params_memory; // wte在buffer的起始位置
	param_tensors.wpe = params_memory + wte_size; // wpe紧跟在wte之后

	return TEE_SUCCESS;
}

static TEE_Result encoder_forward_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_NONE);

	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

	// 解析参数
	int* inputs = (int*)params[0].memref.buffer;
	int B = params[1].value.a;
	int T = params[1].value.b;
	int C = params[2].value.a;
    // 分配输出内存
	int output_size = B * T * C * sizeof(float);
	act_tensors.encoded = malloc(output_size);
	if (!act_tensors.encoded) {
		EMSG("Out of memory");
		return TEE_ERROR_OUT_OF_MEMORY;
	}
	// 计算
	float* out = act_tensors.encoded;
	float* wte = param_tensors.wte;
	float* wpe = param_tensors.wpe;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inputs[b * T + t];
            float* wte_ix = wte + ix * C;
            float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
	return TEE_SUCCESS;
}

static TEE_Result encoder_output_TA(uint32_t param_types, TEE_Param params[4])
{ 
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);
	
	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	// 输出act_tensors.encoded到输出缓冲区
	float *buffer = (float*)params[0].memref.buffer;
	size_t buffer_size = params[0].memref.size;
	memcpy(buffer, act_tensors.encoded, buffer_size);
	return TEE_SUCCESS;
}
// matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
// softmax_forward(acts.probs, acts.logits, B, T, V);
static TEE_Result matmul_softmax_forward_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_VALUE_INPUT,
						   TEE_PARAM_TYPE_NONE);
	
	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	// 解析参数
	float* inputs = (float*)params[0].memref.buffer;
	int B = params[1].value.a;
	int T = params[1].value.b;
	int C = params[2].value.a;
	int V = params[2].value.b;
	// 分配输出内存
	int output_size = B * T * V * sizeof(float);
	act_tensors.logits = malloc(output_size);
	act_tensors.probs = malloc(output_size);
	if (!act_tensors.logits || !act_tensors.probs) {
		EMSG("Out of memory");
		return TEE_ERROR_OUT_OF_MEMORY;
	}
	// 计算
	matmul_forward_TA(act_tensors.logits, inputs, param_tensors.wte, NULL, B, T, C, V);
	softmax_forward_TA(act_tensors.probs, act_tensors.logits, B, T, V);
	return TEE_SUCCESS;
}

static TEE_Result softmax_output_TA(uint32_t param_types, TEE_Param params[4])
{
	uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE,
						   TEE_PARAM_TYPE_NONE);
	
	DMSG("has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;
	// 输出act_tensors.probs到输出缓冲区
	float *buffer = (float*)params[0].memref.buffer;
	size_t buffer_size = params[0].memref.size;
	memcpy(buffer, act_tensors.probs, buffer_size);
	return TEE_SUCCESS;
}
/* 当一个TA被调用时调用。sess_ctx保存由TA_OpenSessionEntryPoint()设置的值。其余的参数来自普通世界 */
TEE_Result TA_InvokeCommandEntryPoint(void __unused *sess_ctx,
				      uint32_t cmd_id, uint32_t param_types,
				      TEE_Param params[4])
{
	switch (cmd_id) {
	case TA_GPT_CMD_LOAD_PARAMS:
		return load_parameters_TA(param_types, params);
	case TA_GPT_CMD_ENCODER_FORWARD:
		return encoder_forward_TA(param_types, params);
	case TA_GPT_CMD_ENCODER_OUTPUT:
		return encoder_output_TA(param_types, params);
	case TA_GPT_CMD_SOFTMAX_FORWARD:
		return matmul_softmax_forward_TA(param_types, params);
	case TA_GPT_CMD_SOFTMAX_OUTPUT:
		return softmax_output_TA(param_types, params);
	default:
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
