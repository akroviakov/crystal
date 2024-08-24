CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
SM   ?= 60

NVCC = nvcc

SM_TARGETS   = -gencode=arch=compute_$(SM),code=\"sm_$(SM),compute_$(SM)\" 
SM_DEF     = -DSM$(SM)0

# non-caching loads (L1 disabled): -Xptxas -dlcm=cg
# caching loads (L1 enabled, default): -Xptxas -dlcm=ca

NVCCFLAGS += --std=c++17 --expt-extended-lambda $(SM_DEF) -Xptxas="-v" -lineinfo -Xcudafe  -\# 

SRC = src
BIN = bin
OBJ = obj

CUB_DIR = cub/

INCLUDES = -I$(CUB_DIR) -I$(CUB_DIR)test -I. -I$(INC)

$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/%: $(OBJ)/%.o
	$(NVCC) $(SM_TARGETS) -lcurand $^ -o $@

setup:
	if [ ! -d "cub"  ]; then \
    wget https://github.com/NVlabs/cub/archive/1.6.4.zip; \
    unzip 1.6.4.zip; \
    mv cub-1.6.4 cub; \
    rm 1.6.4.zip; \
	fi
	mkdir -p bin/ssb obj/ssb
	mkdir -p bin/ops obj/ops

clean:
	rm -rf bin/* obj/*
