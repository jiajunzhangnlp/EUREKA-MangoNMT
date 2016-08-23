### Compilation options.

# C++ compiler. Tested with g++ and Intel icpc.
CXX=/usr/bin/g++
#CXX=/usr/local/bin/clang-omp++
#CXX=/opt/local/bin/g++-mp-4.7
#CXX=icpc
#CXX=nvcc
# Compiler options. Note that -DEIGEN_NO_DEBUG is essential for good performance!
#this should be unset to SHARE_SAMPLES if you want to share the noise samples across the minibatch
#SHARE_FLAG=-DSHARE_SAMPLES
PEEP=-DPEEP
#BOOST_CFLAGS=-g -DDOUBLE $(SHARE_FLAG) #-DPEEP #-DSHARE_SAMPLES 
#CFLAGS=-O3 -DEIGEN_NO_DEBUG -DNDEBUG -DDOUBLE $(SHARE_FLAG)
#CFLAGS=-O3 -DNDEBUG -DDOUBLE $(SHARE_FLAG)
CFLAGS=-O3 -DEIGEN_NO_DEBUG -DNDEBUG $(PEEP) -DFLOAT $(SHARE_FLAG) -DCLIP_SPARSE
#CFLAGS=-g -lefence -DEIGEN_NO_DEBUG -DNDEBUG $(PEEP) -DFLOAT $(SHARE_FLAG) -DCLIP_SPARSE
# Architecture. Set to x86_64 or i686 to override.
ARCH:=$(shell uname -m)
# Operating system. Set to override (the only option that makes any difference is Darwin).
OS:=$(shell uname -s)

# To build static binaries, uncomment the line below:
#STATIC=1

### Required libraries. You must install these prior to building.

# Set this to the root directory of Boost (should have a subdirectory named boost):

BOOST=/usr/local/include/boost
#BOOST=/usr
#BOOST=/opt/local
# Where to find Boost header files
BOOST_INC=$(BOOST)

# Set this to the root directory of Eigen (should have a subdirectory named Eigen):
EIGEN=../3rdparty

### Optional libraries.

# To disable multithreading, comment out the line below:
OMP=1

# To use the MKL library, uncomment the line below and set it to the MKL root:
MKL=/opt/intel/mkl
# Set to 1 if you want to use the Single Dynamic Library; comment out otherwise.
# This is required for building the Python extensions, but doesn't work with building a static binary.
MKL_SINGLE=1

# For Python bindings, set the following and run 'make python/nplm.so'.
PYTHON_VERSION=2.7
PYTHON_ROOT=/opt/local/Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)
CYTHON=$(PYTHON_ROOT)/bin/cython

##### End of configurable options #####

# used for profiling
#USE_CHRONO=1

TCLAP=../3rdparty/tclap/include

# Currently, this is needed only if USE_CHRONO is defined:
# Where to find Boost libraries
BOOST_LIB=/usr/local/lib
# On some systems, a suffix is appended for the multithreaded version.
BOOST_LIB_SUFFIX=
#BOOST_LIB_SUFFIX=-mt

BOOST_CFLAGS=-I$(BOOST_INC)
BOOST_LDFLAGS=
BOOST_LDLIBS=-lboost_iostreams$(BOOST_LIB_SUFFIX) -lboost_system$(BOOST_LIB_SUFFIX) -lboost_filesystem$(BOOST_LIB_SUFFIX)
#BOOST_LDLIBS=-lboost_system$(BOOST_LIB_SUFFIX) -lboost_thread$(BOOST_LIB_SUFFIX)
ifdef USE_CHRONO
  BOOST_CFLAGS+=-DUSE_CHRONO
  BOOST_LDLIBS+=-lboost_system$(BOOST_LIB_SUFFIX) -lboost_chrono$(BOOST_LIB_SUFFIX)
endif
ifdef BOOST_LDLIBS
  BOOST_LDFLAGS+=-L$(BOOST_LIB) -Wl,-rpath -Wl,$(BOOST_LIB)
endif

ifdef OMP
  ifneq (,$(findstring g++,$(CXX)))
    OMP_CFLAGS=-fopenmp
    OMP_LDFLAGS=-fopenmp
  endif
  ifneq (,$(findstring icpc,$(CXX)))
    OMP_CFLAGS=-openmp
    OMP_LDFLAGS=-openmp
  endif
  ifneq (,$(findstring clang-omp++,$(CXX)))
    OMP_CFLAGS=-fopenmp
    OMP_LDFLAGS=-fopenmp
  endif  
endif

ifdef MKL
  ifdef MKL_SINGLE
    ifeq ($(ARCH),x86_64)
      MKL_LDFLAGS=-L$(MKL)/lib/intel64 -Wl,-rpath -Wl,$(MKL)/lib/intel64
    endif
    ifeq ($(ARCH),i686)
      MKL_LDFLAGS=-L$(MKL)/lib/ia32 -Wl,-rpath -Wl,$(MKL)/lib/ia32
    endif
    MKL_CFLAGS=-I$(MKL)/include -DEIGEN_USE_MKL_ALL -DMKL_SINGLE
    MKL_LDLIBS=-lmkl_rt

  else

    MKL_CFLAGS=-I$(MKL)/include -DEIGEN_USE_MKL_ALL
    MKL_LDLIBS=-Wl,--start-group
    ifeq ($(ARCH),x86_64)
      MKL_LDFLAGS=-L$(MKL)/lib/intel64 -Wl,-rpath -Wl,$(MKL)/lib/intel64
      MKL_LDLIBS+=-lmkl_intel_lp64
    endif
    ifeq ($(ARCH),i686)
      MKL_LDFLAGS=-L$(MKL)/lib/ia32 -Wl,-rpath -Wl,$(MKL)/lib/ia32
      MKL_LDLIBS+=-lmkl_intel
    endif

    ifneq (,$(findstring g++,$(CXX)))
      MKL_LDLIBS+=-lmkl_gnu_thread
    endif
    ifneq (,$(findstring icpc,$(CXX)))
      MKL_LDLIBS+=-lmkl_intel_thread
    endif

    MKL_LDLIBS+=-lmkl_core -Wl,--end-group
endif
endif


ifdef STATIC
  LDFLAGS+=-static
endif

ALL_CFLAGS=$(OMP_CFLAGS) $(MKL_CFLAGS) $(BOOST_CFLAGS) -I$(TCLAP) -I$(EIGEN) $(CFLAGS)
ALL_LDFLAGS=$(OMP_LDFLAGS) $(MKL_LDFLAGS) $(BOOST_LDFLAGS) $(LDFLAGS)
ALL_LDLIBS=$(MKL_LDLIBS) $(BOOST_LDLIBS)

PYTHON_CFLAGS+=-I$(PYTHON_ROOT)/include/python$(PYTHON_VERSION)
ifeq ($(OS),Darwin)
  # avoid having to link in libpython
  PYTHON_LDFLAGS+=-undefined dynamic_lookup
endif

# Some other programs

AR=ar
RANLIB=ranlib

# Rules

BINS=trainNeuralNetwork generateFromNetwork generateFromNetworkAdapt
OBJS=util.o model.o

all: $(BINS) 

clean:
	rm -f *.o shared/*.o python/*.o $(BINS) $(LIBS) python/nplm.{cpp,so} python/nptm.{cpp,so}

install: all
	mkdir -p ../bin
	cp $(BINS) ../bin
	mkdir -p ../lib
	cp $(LIBS) ../lib

%.o: %.cpp
	$(CXX) -c $(ALL_CFLAGS) $< -o $@

shared/%.o: %.cpp
	$(CXX) -c -fPIC $(ALL_CFLAGS) $< -o $@

trainNeuralNetwork: trainNeuralNetwork.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@
	
generateFromNetwork: generateFromNetwork.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@		
	
generateFromNetworkAdapt: generateFromNetworkAdapt.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@		
	
	



%.cpp: %.pyx
	$(CYTHON) --cplus $^ -o $@


