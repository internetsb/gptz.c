# C语言GPT2推理仓库

这是一个纯C语言的gpt2推理仓库，旨在在有限资源下推理，以及原理学习。

原作者：https://git.nju.edu.cn/jyy/os2025/-/tree/M6

## 如何开始？

1. 首先需要下载gpt2模型
   ```bash
   python3 download_model.py
   ```

2. 然后导出模型BPE
   ```bash
   python3 export_tokenizer.py
   ```

前两部将创建models/gpt2_124M.bin及models/gpt2_ranks.bin

3. 编译代码
   ```bash
   make all
   ```

## 可执行程序说明：

- `gpt`：接收token输入，返回token输出
- `cchat`：输入自然语言->转换为token序列->调用gpt->将输出token转为自然语言

## 示例：

```
./cchat
Text to complete: Ladies and # 输入
Gentlemen, this year's NBA # 模型补全
```

## 许可证：
MIT License