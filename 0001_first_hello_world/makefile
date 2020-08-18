CC = nvcc
CFLAGS = -Wall

BUILD_DIR := ./build

SOURCE=$(wildcard *.cu)
EXECS=$(patsubst %.cu,%,$(SOURCE))  

all: $(EXECS)

% : %.cu
	$(CC) -o $(BUILD_DIR)/$@ $<



.PHONY:clean  
clean:
	-rm -f build/*