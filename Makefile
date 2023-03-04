.PHONY: all build debug clean profile bench cuobjdump

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

bench: build
	@bash gen_benchmark_results.sh
