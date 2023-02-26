.PHONY: all build clean profile bench

CMAKE := cmake
NINJA := ninja

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. -G Ninja
	@$(NINJA) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

bench: build
	@bash gen_benchmark_results.sh
