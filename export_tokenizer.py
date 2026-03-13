import tiktoken
import struct
import os

enc = tiktoken.get_encoding("gpt2")
# gpt2 的 ranks 是 dict[bytes, int]
ranks = enc._mergeable_ranks

# 确保 models 目录存在
os.makedirs("models", exist_ok=True)

# 检查文件是否已存在
output_file = "models/gpt2_ranks.bin"
if os.path.exists(output_file):
    response = input(f"文件 {output_file} 已存在，是否要覆盖? (y/N): ")
    if response.lower() != 'y':
        print("取消导出。")
        exit()

with open(output_file, "wb") as f:
    for token, rank in ranks.items():
        # 格式：[1字节长度][N字节内容][4字节Rank(int)]
        f.write(struct.pack("B", len(token)))
        f.write(token)
        f.write(struct.pack("i", rank))

print(f"Tokenizer 已成功导出到 {output_file}")