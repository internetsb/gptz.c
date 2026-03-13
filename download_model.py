#!/usr/bin/env python3
"""
下载GPT模型文件的脚本
"""

import os
import urllib.request
import sys

def download_model():
    """从指定URL下载模型文件到models目录"""
    url = "https://box.nju.edu.cn/f/ca18466ad6054f2d85c3/?dl=1"
    model_filename = "gpt2_124M.bin"
    target_dir = "models"
    
    # 确保models目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, model_filename)
    
    print(f"正在从 {url} 下载模型文件...")
    print(f"目标路径: {target_path}")
    
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"模型文件已成功下载到: {target_path}")
    except Exception as e:
        print(f"下载失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if os.path.exists("models/gpt2_124M.bin"):
        print("模型文件已存在，无需下载")
    else:
        print("模型文件不存在，开始下载...")
        download_model()