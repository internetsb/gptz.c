NAME := $(shell basename $(PWD))
export MODULE := M6

# 编译器和编译选项
CC ?= gcc
CFLAGS += -O2 -Wall -Wextra -std=c11 -I./includes
LIBS = -lm -lpthread

# 源文件和对应的目标文件
SOURCES = cchat.c gpt.c
TARGETS = cchat gpt

# 默认目标：构建所有程序
all: $(TARGETS)

# 单独构建每个程序
cchat: cchat.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

gpt: gpt.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# 清理生成的文件
clean:
	rm -f $(TARGETS)

# 重新构建所有
rebuild: clean all

.PHONY: all clean rebuild