
DOUBLEPRECISION=-DDOUBLE_PRECISION
all:
	nvcc $(DOUBLEPRECISION) -I uammd/src -I uammd/src/third_party -O3 -std=c++14 fcm_example.cu -o fcm -lcufft
