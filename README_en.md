# GPT2 Inference Repository in C

>[!WARNING]
>This repository is for learning purposes regarding OP-TEE and LLM principles. Please do not use directly in production environments.

**[[简体中文]](README.md)**

This is a simple, unoptimized GPT2 inference repository for the OP-TEE environment developed for educational purposes.

## How to Get Started?

1. Place this repository under optee_examples
   ```bash
   git clone https://github.com/internetsb/gptz.c.git
   ```

2. Download the GPT2 model
   ```bash
   python3 download_model.py
   ```

3. Export the model BPE tokenizer, requires tiktoken library
   ```bash
   python3 export_tokenizer.py
   ```

Steps 2 and 3 will create a models folder in the repository directory and download models/gpt2_124M.bin and export models/gpt2_ranks.bin. Since the model is quite large (~500MB) and cannot be placed in out-br/target/root like darknetz, we place the model files in the shared directory between the host and QEMU, and mount it to the virtual machine. (For production environments, pay attention.)

4. Mount the shared directory

   Modify `optee/build/common.mk`
   ```
   #Add a new parameter under the original QEMU_EXTRA_ARGS
   QEMU_EXTRA_ARGS +=\
	-object rng-random,filename=/dev/urandom,id=rng0 \
	-device virtio-rng-pci,rng=rng0,max-bytes=1024,period=1000
   # Add the following new parameters, replace with your actual path
   QEMU_EXTRA_ARGS += -fsdev local,id=fsdev0,path=/home/internetsb/optee/build/shared_dir,security_model=none \
                     -device virtio-9p-device,fsdev=fsdev0,mount_tag=host_share
   ```
   `path=/home/internetsb/optee/build/shared_dir`, this is the directory on my host machine (you need to manually create shared_dir and replace with your real path).

   `mount_tag=host_share`, this is the label for the shared directory.

   Then copy the model files downloaded in steps 2 and 3 to the shared directory.

5. Compile and start OP-TEE
  ```bash
  make run -j$(nproc)
  ```

6. Mount the shared directory in the Normal World
  ```bash
  mkdir /mnt/shared
  mount -t 9p -o trans=virtio host_share /mnt/shared
  ```
  The folder `/mnt/shared` is the shared directory, `ls /mnt/shared` should show the model files.

7. Run the program
  ```bash
  gpt /mnt/shared/models/gpt2_124M.bin /mnt/shared/models/gpt2_ranks.bin -T 2
  ```
  Loading parameters, and putting the final layernorm into TA, you will see output similar to the following:
  ```
  Session started.
  Trusted layer flag: 2
  Parameter 0: 154389504 bytes
  #......
  Loaded parameter 12...
  Loaded parameter 13...
  Loaded parameters into TEE...
  Loaded 497759232 bytes of model parameters
  Model loaded from: /mnt/shared/models/gpt2_124M.bin
  Tokenizer loaded from: /mnt/shared/models/gpt2_ranks.bin
  Text to complete: Ladies and #Text to be completed
  #Model output
  Generated:  Gentlemen, this year's NBA playoffs have never been about making your living from Twitter. But this year is about living with it. We
  #......
  ```
  You can modify the coin parameter in the sample_mult function in host/main.c to change the randomness of the model-generated text.

8. Expanding TA Heap Memory

In step 7, the example command uses the parameter `-T 2`, which means loading normalization layer parameters into TA for inference (to be honest, what's the value in this?).

Below are the steps I actually tested that can expand TA heap memory in the QEMU environment, allowing using parameter `-T 1` to protect the Embedding layer parameters: (Additionally, `-T 0` means all parameters are inferred in the normal world):

```
/* 1. Modify optee_os/core/mm/pgt_cache.c */
// #define PGT_CACHE_SIZE	ROUNDUP(CFG_PGT_CACHE_ENTRIES, PGT_NUM_PGT_PER_PAGE)
#define PGT_CACHE_SIZE 512

/* 2. Modify trusted-firmware-a/plat/qemu/qemu/include/platform_def.h */
#define SEC_DRAM_BASE           0x70000000   // Base address
#define SEC_DRAM_SIZE           0x0C800000	 // Size 200MB

/* 3. Add a line in optee_os/core/arch/arm/plat-vexpress/platform_config.h */
#define TEE_RAM_VA_SIZE (200 * 1024 * 1024) //200MB

/* 4. Modify optee_os/core/arch/arm/plat-vexpress/conf.mk to configure qemu_virt */
CFG_TZDRAM_START ?= 0x70000000 # Don't conflict with Kernel (0x42200000)
CFG_TZDRAM_SIZE  ?= 0x0C800000 # 200M

/* 5. Add a line at the end of build/qemu_v8.mk file */
OPTEE_OS_COMMON_FLAGS += CFG_TZDRAM_START=0x70000000 CFG_TZDRAM_SIZE=0x0C800000

/* 6. Modify optee_examples/gpt/ta/user_ta_header_defines.h */
#define TA_DATA_SIZE			(165 * 1024 * 1024) // 165M heap memory
```

After recompiling and running, use parameter `-T 1` for inference, you will see output similar to the following:
```
  Session started.
  Trusted layer flag: 1
  Parameter 0: 154389504 bytes
  #......
  Parameter 15: 3072 bytes
  Allocated 497759232 bytes for model parameters
  Uploading parameters: 100.0%
  Done.
  Loaded parameters into TEE...
  Loaded parameter 2...
  #......
  Model loaded from: /mnt/shared/models/gpt2_124M.bin
  Tokenizer loaded from: /mnt/shared/models/gpt2_ranks.bin
  Text to complete: Ladies and

  Generated:  Gentlemen, this year's NBA playoffs have
  #......
```

## Random Thoughts

**Heap Memory Limitations**

Initially, this repository intended to load the first encoder and the final matmul and softmax layers into the TA to protect token embedding parameters, but TA memory limitations require tedious modifications and are unstable, so I'm not sure if this is a good idea. Additionally, encrypting the model in the normal world, loading it piece by piece into the secure world for decryption and inference was not considered due to efficiency reasons. 

The model may crash after running for some time. 

**OP-TEE protecting Large Language Models?**

Limited effectiveness:

LLMs require too much memory. Even loading just one layer may fail. On resource-constrained devices, this would be worse. For example, the GPT-2 model used in this repository has a Token Embedding parameter size of approximately 150MB, Attention QKV occupies about 80MB, FFN Weight 1 takes about 110MB. These important parameters are difficult to fit into the limited TEE memory of OP-TEE. And GPT-2 is just a "small" large model; what about models with dozens of billions of parameters?

## References
- [llm.c](https://github.com/karpathy/llm.c)
- [darknetz](https://github.com/mofanv/darknetz)
- [gpt.c](https://git.nju.edu.cn/jyy/os2025/-/tree/M6)

## License:
MIT License