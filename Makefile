CMAKE := cmake
NINJA := ninja

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. -G Ninja
	@$(NINJA) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile:
	@cd $(BUILD_DIR) && $(NINJA) -v
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)
