/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (c) 2016-2017, Linaro Limited
 * All rights reserved.
 */
#ifndef TA_GPT_H
#define TA_GPT_H

#define TA_GPT_UUID \
	{ 0x395c3b47, 0x1fc8, 0x4443, \
		{ 0x9d, 0x99, 0xb8, 0xcf, 0x6d, 0xee, 0x99, 0x7c} }

/* 定义命令ID */
#define TA_GPT_CMD_LOAD_PARAMS 1
#define TA_GPT_CMD_ENCODER_FORWARD 2
#define TA_GPT_CMD_SOFTMAX_FORWARD 3
#define TA_GPT_CMD_LOAD_LNFWB 4
#define TA_GPT_CMD_LAYERNORM_FORWARD 5

#endif /* TA_GPT_H */
