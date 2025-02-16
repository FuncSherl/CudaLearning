BUILD_DIR = build
MAKEFILE_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
INCLUDE_DIR = $(MAKEFILE_PATH)/..

CFLAGS_GPU = -g -std=c++14 -I$(INCLUDE_DIR)
CC_GPU = nvcc
SOURCE_GPU = $(wildcard *.cu)
EXECS_GPU = $(patsubst %.cu,$(BUILD_DIR)/%,$(SOURCE_GPU))

CFLAGS_CPU = -Wall -g -std=c++14 -I$(INCLUDE_DIR)
CC_CPU = g++
SOURCE_CPU = $(wildcard *.cpp)
EXECS_CPU = $(patsubst %.cpp,$(BUILD_DIR)/%,$(SOURCE_CPU))

# $@ -- target file, $^ -- all dependencies, $< -- first dependency.

all: $(EXECS_GPU) $(EXECS_CPU)

show:
	@echo "EXECS_GPU: " $(EXECS_GPU)
	@echo "EXECS_CPU: " $(EXECS_CPU)

$(EXECS_GPU): $(SOURCE_GPU)
	@echo + CC $<
	@mkdir -p $(BUILD_DIR)
	$(CC_GPU) -o $@ $< $(CFLAGS_GPU)

$(EXECS_CPU): $(SOURCE_CPU)
	@echo + CC $<
	@mkdir -p $(BUILD_DIR)
	$(CC_CPU) -o $@ $< $(CFLAGS_CPU)

# Don't remove *.o files automatically
# .SECONDARY: $(OBJS)

.PHONY: clean all show
clean:
	-rm -rf $(BUILD_DIR)
