// Matrix multiplication kernel
// multiplies two matrix_fp32 data arrays into a third output data array
// Worksize should be the matrix width * matrix height (one work item per cell)
const char *matrix_multiply_cl = "                                            \n\
__kernel void matrix_multiply(                                                \n\
    __constant float *array1,                                                 \n\
    __constant float *array2,                                                 \n\
    __global   float *output,                                                 \n\
    const        int  shared_dim,                                             \n\
    const        int  width)                                                  \n\
{                                                                             \n\
    float  tmp = 0;                                                           \n\
    size_t idx = get_global_id(0);                                            \n\
    uint   x   = idx % width;                                                 \n\
    uint   y   = idx / width;                                                 \n\
    uint i;                                                                   \n\
    for (i = 0; i < shared_dim; i++) {                                        \n\
        tmp += array1[y * shared_dim + i] * array2[i * width + x];            \n\
        output[idx] = tmp;                                                    \n\
    }                                                                         \n\
}                                                                             \n\
";
