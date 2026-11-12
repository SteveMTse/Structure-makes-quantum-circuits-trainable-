# =============================================================================
#  Makefile for sun_experiment.cu   (GNU make; works with MSVC and GCC/Clang)
#
#    make                 FP32 build (recommended: ~30x faster on GeForce parts,
#                         whose FP64 rate is 1/64 of FP32)
#    make PREC=fp64       FP64 build
#    make ARCH="-gencode arch=compute_89,code=sm_89"
#    make HOST=gnu | HOST=msvc
#    make config          show what was detected
# =============================================================================

NVCC ?= nvcc
SRC   = sun_experiment.cu
BIN   = sun_experiment
PREC ?= fp32
TPB  ?= 256

# ---- shell detection (cmd.exe echoes the quotes back; POSIX shells strip) ---
ifeq ($(shell echo "x"),"x")
  WINCMD := 1
else
  WINCMD := 0
endif

ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
else
  EXEEXT :=
endif
TARGET := $(BIN)$(EXEEXT)

# ---- host compiler flag style ----------------------------------------------
ifeq ($(OS),Windows_NT)
  HOST ?= msvc
else
  HOST ?= gnu
endif

ifeq ($(HOST),msvc)
  # cl.exe takes "-O2" as well as "/O2"; the dash form survives MSYS/Git Bash
  # path rewriting. MSVC has threading built in, so no -pthread.
  HOSTFLAGS := -Xcompiler "-O2"
else
  HOSTFLAGS := -Xcompiler "-O3 -pthread"
endif

ifneq ($(CCBIN),)
  HOSTFLAGS += -ccbin "$(CCBIN)"
endif

# ---- precision --------------------------------------------------------------
ifeq ($(PREC),fp32)
  PRECFLAG := -DSUN_FP32
else ifeq ($(PREC),fp64)
  PRECFLAG :=
else
  $(error PREC must be fp32 or fp64)
endif

# ---- GPU architecture -------------------------------------------------------
NVCC_VER := $(patsubst V%,%,$(filter V%,$(shell $(NVCC) --version)))
NVCC_MAJ := $(word 1,$(subst ., ,$(NVCC_VER)))
NVCC_MIN := $(word 2,$(subst ., ,$(NVCC_VER)))

ARCH_FALLBACK := -gencode arch=compute_70,code=sm_70 \
                 -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86

ifeq ($(NVCC_MAJ),10)
  ARCH ?= $(ARCH_FALLBACK)
else ifeq ($(NVCC_MAJ),11)
  ifeq ($(filter 5 6 7 8 9,$(NVCC_MIN)),)
    ARCH ?= $(ARCH_FALLBACK)
  else
    ARCH ?= -arch=native
  endif
else
  ARCH ?= -arch=native
endif

# No --use_fast_math: the numbers are the point. No cuBLAS/cuSOLVER any more.
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -DTPB=$(TPB) $(PRECFLAG) $(HOSTFLAGS)

ifeq ($(WINCMD),1)
  RM_F := del /Q
else
  RM_F := rm -f
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	-$(RM_F) $(TARGET)

config:
	@echo "nvcc      = $(NVCC)  (version $(NVCC_VER))"
	@echo "host      = $(HOST)"
	@echo "shell     = $(if $(filter 1,$(WINCMD)),cmd.exe,posix)"
	@echo "target    = $(TARGET)"
	@echo "arch      = $(ARCH)"
	@echo "precision = $(PREC)"
	@echo "hostflags = $(HOSTFLAGS)"

.PHONY: all clean config
