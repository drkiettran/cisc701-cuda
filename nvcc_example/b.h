//---------- b.h ----------
#ifndef __B_H__
#define __B_H__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#define N 8
extern __device__ int g[N];
extern __device__ void bar(void);
#endif