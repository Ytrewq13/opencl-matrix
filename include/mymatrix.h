#ifndef INT_MAX
#include <limits.h>
#endif
#ifndef INFINITY
#include <math.h>
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    float *data;
    int width, height;
} matrix_fp32;

enum {
    ALLOCATION_FAILURE  = -1,
    INVALID_SHAPES      = -2,
    INVALID_MATRIX      = -3
};

void print_matrix(matrix_fp32 *m);
int create_matrix_fp32(size_t h, size_t w, float *d, matrix_fp32 **result);
int transpose(matrix_fp32 *m, matrix_fp32 **result);
float determinant(matrix_fp32 *m, int *err);
int mat_fp32_multiply(matrix_fp32 *a, matrix_fp32 *b, matrix_fp32 **result);
