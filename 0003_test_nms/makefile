

BUILD_DIR = build

CFLAGS_GPU= -g -std=c++14 `pkg-config --cflags --libs opencv`
CC_GPU = nvcc
SOURCE_GPU=$(wildcard *.cu) 
EXECS_GPU=$(patsubst %.cu,$(BUILD_DIR)/%,$(SOURCE_GPU))  #$(SOURCE_GPU:%.cu=%)

CFLAGS_CPU=-Wall -g -std=c++14 `pkg-config --cflags --libs opencv`
CC_CPU=g++
SOURCE_CPU=$(wildcard *.cpp) 
EXECS_CPU=$(patsubst %.cpp,$(BUILD_DIR)/%,$(SOURCE_CPU))  #$(SOURCE_GPU:%.cpp=%)


#$@--目标文件，$^--所有的依赖文件，$<--第一个依赖文件。

all: $(EXECS_GPU) $(EXECS_CPU)


show:
	@echo "EXECS_GPU: " $(EXECS_GPU)
	@echo "EXECS_CPU: " $(EXECS_CPU)

$(EXECS_GPU) : $(SOURCE_GPU)
	@echo + CC $<
	@mkdir -p $(BUILD_DIR)
	$(CC_GPU) -o $@ $< $(CFLAGS_GPU)

$(EXECS_CPU) : $(SOURCE_CPU)
	@echo + CC $<
	@mkdir -p $(BUILD_DIR)
	$(CC_CPU) -o $@ $< $(CFLAGS_CPU)

# Don't remove *.o files automatically
# .SECONDARY: $(OBJS)

.PHONY:clean  all show
clean:
	-rm -rf $(BUILD_DIR)
