//---------- b.cu ----------
#include "b.h"

__device__ int g[N];
__device__ void bar(void)
{
    g[threadIdx.x]++;
    printf("Kernel 'bar' exits.\n");
}