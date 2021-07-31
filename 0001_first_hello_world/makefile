CC = nvcc
CFLAGS = -Wall

BUILD_DIR = build

SOURCE=$(wildcard *.cu)
EXECS=$(patsubst %.cu,$(BUILD_DIR)/%,$(SOURCE))  

all: $(EXECS)

$(BUILD_DIR)/% : %.cu
	@echo + CC $<
	@mkdir -p $(BUILD_DIR)
	$(CC) -g -o $@ $<

# Don't remove *.o files automatically
# .SECONDARY: $(OBJS)

.PHONY:clean  all
clean:
	-rm -rf $(BUILD_DIR)