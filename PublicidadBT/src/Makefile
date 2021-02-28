CUDA_INSTALL_DIR?=/usr/local/cuda
CUDNN_INSTALL_DIR?=$(CUDA_INSTALL_DIR)
CUDA_LIBDIR=lib
CUDNN_LIBDIR=lib64
COMMON_LIBS = -lcudnn -lcublas -lcudart -lrt -ldl -lpthread
LIBS  =-lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser $(COMMON_LIBS)
INCPATHS += -I"../util" -I"$(CUDA_INSTALL_DIR)/include" -I"$(CUDNN_INSTALL_DIR)/include"
LIBPATHS += -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)" -L"$(CUDNN_INSTALL_DIR)/$(CUDNN_LIBDIR)"
OPENCV = `pkg-config --libs opencv4 --cflags opencv4`
COMMON_FLAGS += -Wall -Wno-deprecated-declarations -std=c++11 $(INCPATHS)
COMMON_FLAGS += -D_REENTRANT
CFLAGS=$(COMMON_FLAGS)
COMMON_LD_FLAGS += $(LIBPATHS)
LFLAGS=$(COMMON_LD_FLAGS)

run: main.cpp VideoPredefinido.o DetectorRostros.o logger.o
	g++ main.cpp VideoPredefinido.o DetectorRostros.o logger.o $(CFLAGS) $(LFLAGS) $(LIBS) $(OPENCV) -o main
	rm -r *.o
	
VideoPredefinido.o: ./camera/VideoPredefinido.cpp
	g++ ./camera/VideoPredefinido.cpp $(OPENCV) -c

DetectorRostros.o: ./rna/DetectorRostros.cpp
	g++ ./rna/DetectorRostros.cpp $(CFLAGS) $(LFLAGS) $(LIBS) $(OPENCV) -c

logger.o: ./util/logger.cpp
	g++ ./util/logger.cpp $(CFLAGS) $(LFLAGS) $(LIBS) -c

clean:
	rm -r *.o