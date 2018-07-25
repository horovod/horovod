ROOTDIR = $(CURDIR)

ifndef config
ifneq ("$(wildcard ./config.mk)", "")
        config = config.mk
else
        config = make/config.mk
endif
endif
include $(config)

HOROVOD_PATH = $(ROOTDIR)/horovod
MXNET_PATH = ../..
DMLC_PATH = ../dmlc-core
MSHADOW_PATH = ../mshadow

include $(MSHADOW_PATH)/make/mshadow.mk

export LDFLAGS = -pthread -lm
export CFLAGS = -std=c++11 -Wall -O2 -fPIC
INCPATH = -I$(HOROVOD_PATH)/common -I$(HOROVOD_PATH)/mxnet -I$(MXNET_PATH)/include -I$(DMLC_PATH)/include -I$(MSHADOW_PATH)
CFLAGS += $(INCPATH)

ifneq ($(ADD_CFLAGS), NONE)
        CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
        LDFLAGS += $(ADD_LDFLAGS)
endif

# specify tensor path
.PHONY: clean all

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
        SHARED_LIBRARY_SUFFIX := dylib
        WHOLE_ARCH= -all_load
        NO_WHOLE_ARCH= -noall_load
        LDFLAGS += -undefined dynamic_lookup
else
        SHARED_LIBRARY_SUFFIX := so
        WHOLE_ARCH= --whole-archive
        NO_WHOLE_ARCH= --no-whole-archive
endif

ifeq ($(USE_CUDA), 1)
        CFLAGS += -I$(ROOTDIR)/3rdparty/cub -DHAVE_CUDA=1 -I$(USE_CUDA_PATH)/include
        ALL_DEP += $(CUOBJ) $(EXTRA_CUOBJ) $(PLUGIN_CUOBJ)
        LDFLAGS += -lcufft
        ifeq ($(ENABLE_CUDA_RTC), 1)
                LDFLAGS += -lcuda -lnvrtc
                CFLAGS += -DMXNET_ENABLE_CUDA_RTC=1
        endif
        # Make sure to add stubs as fallback in order to be able to build
        # without full CUDA install (especially if run without nvidia-docker)
        LDFLAGS += -L/usr/local/cuda/lib64/stubs
        SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-gpu
        ifeq ($(USE_NCCL), 1)
                ifneq ($(USE_NCCL_PATH), NONE)
                        CFLAGS += -I$(USE_NCCL_PATH)/include
                        LDFLAGS += -L$(USE_NCCL_PATH)/lib
                endif
                LDFLAGS += -lnccl
                CFLAGS += -DHAVE_NCCL=1
        else
                CFLAGS += -DHAVE_NCCL=0
        endif

        ifeq ($(HOROVOD_GPU_ALLREDUCE), MPI)
                HOROVOD_GPU_ALLREDUCE = M
        else
                HOROVOD_GPU_ALLREDUCE = N
        endif

        ifeq ($(HOROVOD_GPU_ALLGATHER), MPI)
                HOROVOD_GPU_ALLGATHER = M
        else
                HOROVOD_GPU_ALLGATHER = N
        endif

        ifeq ($(HOROVOD_GPU_BROADCAST), MPI)
                HOROVOD_GPU_BROADCAST = M
        else
                HOROVOD_GPU_BROADCAST = N
        endif

	ifeq ($(USE_MPI_PATH),)
  	# Default mpi
		USE_MPI_PATH := $(shell ./prepare_mpi.sh $(DEF_MPI_PATH))
	endif

        CFLAGS += -I$(USE_MPI_PATH)/include
        LDFLAGS += -L$(USE_MPI_PATH)/lib -Wl,-rpath=$(USE_MPI_PATH)/lib -lmpi

else
        CFLAGS += -DHAVE_NCCL=0 -DHAVE_CUDA=0
endif

all: lib/libhorovod.a

ALL_OBJ = $(addprefix build/common/, common.o mpi_message.o operations.o timeline.o)
ALL_OBJ += $(addprefix build/mxnet/, adapter.o cuda_util.o handle_manager.o mpi_ops.o ready_event.o tensor_util.o)
ALL_DEP = $(ALL_OBJ)

lib/libhorovod.a: $(ALL_DEP)
	@mkdir -p $(@D)
	${AR} crv $@ $(filter %.o, $?)

build/common/%.o: horovod/common/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -std=c++11 -MM -MT build/common/$*.o $< >build/common/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/mxnet/%.o: horovod/mxnet/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -std=c++11 -MM -MT build/mxnet/$*.o $< >build/mxnet/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build lib
