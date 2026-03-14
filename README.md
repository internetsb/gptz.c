# C语言GPT2推理仓库
>[!WARNING]
>该仓库仅供optee及llm原理学习，请勿直接用于生产环境。

这是一个简单、未充分优化的optee环境的gpt2推理仓库，出于学习目的开发。

## 如何开始？

1. 将本仓库放入optee_examples下
   ```bash
   git clone https://github.com/internetsb/gptz.c.git
   ```

2. 下载gpt2模型
   ```bash
   python3 download_model.py
   ```

3. 然后导出模型BPE
   ```bash
   python3 export_tokenizer.py
   ```

第2,3步将在仓库目录下创建models文件夹，并下载models/gpt2_124M.bin及导出models/gpt2_ranks.bin，由于模型过大（约500MB），无法像darknetz一样放入out-br/target/root，我们将模型文件放入宿主机与qemu的共享目录，并挂载到虚拟机。

4. 挂载共享目录

   修改`optee/build/common.mk`
   ```
   #在原先的QEMU_EXTRA_ARGS下新增一段参数
   QEMU_EXTRA_ARGS +=\
	-object rng-random,filename=/dev/urandom,id=rng0 \
	-device virtio-rng-pci,rng=rng0,max-bytes=1024,period=1000
   # 以下为新增参数，替换成你的真实路径
   QEMU_EXTRA_ARGS += -fsdev local,id=fsdev0,path=/home/internetsb/optee/build/shared_dir,security_model=none \
                     -device virtio-9p-device,fsdev=fsdev0,mount_tag=host_share
   ```
   `path=/home/internetsb/optee/build/shared_dir`，这是我宿主机上的目录（需要手动创建shared_dir并替换为你的真实路径）。

   `mount_tag=host_share`，这是共享目录的标签。

   然后将第2，3步下载的模型文件复制到共享目录下

5. 编译启动op-tee
  ```bash
  make run -j$(nproc)
  ```

6. 在 Normal World 挂载共享目录
  ```bash
  mkdir /mnt/shared
  mount -t 9p -o trans=virtio host_share /mnt/shared
  ```
  文件夹`/mnt/shared`即为共享目录，`ls /mnt/shared`应能看到模型文件。

7. 运行程序
  ```bash
  gpt /mnt/shared/models/gpt2_124M.bin /mnt/shared/models/gpt2_ranks.bin -T 2
  ```
  你将看到类似于以下的输出：
  ```
  Session started.
  Trusted layer flag: 2
  Parameter 0: 154389504 bytes
  ...
  Loaded parameter 12...
  Loaded parameter 13...
  Loaded parameters into TEE...
  Loaded 497759232 bytes of model parameters
  Model loaded from: /mnt/shared/models/gpt2_124M.bin
  Tokenizer loaded from: /mnt/shared/models/gpt2_ranks.bin
  Text to complete: Ladies and #待补全文本
  #模型输出
  Generated:  Gentlemen, this year's NBA playoffs have never been about making your living from Twitter. But this year is about living with it. We
  ......
  ```
  你可以修改host/main.c中sample_mult函数的coin参数来改变模型生成文本的随机性。
## 杂谈

**堆内存限制**
本仓库一开始想将第一层encoder与最后一层matmul和softmax加载入TA以保护词嵌入参数，但TA的内存限制修改繁琐且不稳定，最终这部分代码未测试但保留，你可以使用 -T 1 看到安全世界中失败的内存分配，在普通世界加密模型，分块载入安全世界解密并推理的做法因效率原因未考虑。

**OP-TEE保护大模型？**
作用有限：
大模型内存占用过大，即使仅放入一层也有可能失败，在资源受限的设备上，这样会更糟糕，例如本仓库使用的gpt-2模型，Token Embedding 参数大小约150MB， Attention QKV约占用80MB,FFN Weight 1约110MB，这些重要参数难以放入OP-TEE有限的TEE内存中。

## 参考
- [llm.c](https://github.com/karpathy/llm.c)
- [darknetz](https://github.com/mofanv/darknetz)
- [gpt.c](https://git.nju.edu.cn/jyy/os2025/-/tree/M6)

## 许可证：
MIT License