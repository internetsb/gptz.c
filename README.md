# C语言GPT2推理仓库

这是一个简单、未充分优化的optee环境的gpt2推理仓库，旨在学习目的。

原作者：https://github.com/karpathy/llm.c

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

>[!WARNING]
>若后续转移到开发板，需要额外注意文件大小这一点！

4. 挂载共享目录

   修改`optee/build/common.mk`
   ```
   #在原先的QEMU_EXTRA_ARGS下新增一段参数
   QEMU_EXTRA_ARGS +=\
	-object rng-random,filename=/dev/urandom,id=rng0 \
	-device virtio-rng-pci,rng=rng0,max-bytes=1024,period=1000
   # 以下为新增参数
   QEMU_EXTRA_ARGS += -fsdev local,id=fsdev0,path=/home/internetsb/optee/build/shared_dir,security_model=none \
                     -device virtio-9p-device,fsdev=fsdev0,mount_tag=host_share
   ```
   `path=./shared_dir`，这是宿主机上的目录（需要手动创建：`mkdir optee/build/shared_dir`）。
   `mount_tag=host_share`，这是共享目录的标签。

5. 编译启动op-tee
  ```bash
  make run -j$(nproc)
  ```

6. 挂载共享目录
  ```bash
  mkdir /mnt/shared
  mount -t 9p -o trans=virtio host_share /mnt/shared
  ```
  文件夹`/mnt/shared`即为共享目录。

7. 运行程序
  ```bash
  gpt /mnt/shared/models/gpt2_124M.bin /mnt/shared/models/gpt2_ranks.bin
  ```
  你将看到类似于以下的输出：
  ```
  Model loaded from: /mnt/shared/models/gpt2_124M.bin
  Tokenizer loaded from: /mnt/shared/models/gpt2_ranks.bin
  Text to complete: Ladies and

  Generated:  Gentlemen, this year's NBA playoffs have never been about making your living from Twitter. But this year is about living with it. We
  ......
  ```

  可喜可贺，可喜可贺。

## 许可证：
MIT License