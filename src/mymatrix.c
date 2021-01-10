#include <stdio.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include "mymatrix.h"

/* mymatrix.c
 * Copyright Sam Whitehead, 2021
 * Last updated 2021-01-10
 */


void print_matrix(matrix_fp32 *m)
{
    int i, j;
    for (i = 0; i < m->height; i++)
    {
        printf("[");
        for (j = 0; j < m->width; j++)
        {
            printf("%8.3f", m->data[i*m->width+j]);
            if (j < m->width-1) printf(" ");
        }
        printf("]\n");
    }
}

// Allocates a matrix_fp32 struct and assigns it to the final argument.
// REMEMBER TO FREE() THE RESULT MATRIX!!!
// Arguments:
// h - The height of the matrix
// w - The width of the matrix
// d - An array of floats containing the data that will be inserted into the
//     matrix
// result - A pointer that is modified to point to the matrix_fp32 struct that
//          was allocated for the matrix. If this is non-null then this function
//          assumes that the address pointed to contains enough space for the
//          matrix struct.
// Return value:
// - 0 if there was no error
// - 1 if there was an error
int create_matrix_fp32(size_t h, size_t w, float *d, matrix_fp32 **result)
{
    if (*result == NULL)
        *result = (matrix_fp32 *)malloc(sizeof(matrix_fp32));
    if (*result == NULL) return 1;
    if (d == NULL)
        d = (float *)malloc(sizeof(float)*w*h);
    if (d == NULL) return 1;
    (*result)->height = h;
    (*result)->width = w;
    (*result)->data = d;
    return 0;
}

// Transpose a matrix into a given pointer location.
// Arguments:
// m - matrix to greate the transpose of
// result - A pointer that is modified to point to the matrix_fp32 struct that
//          was allocated for the transpose
// Return value:
// - 0 if there was no error
// - 1 if there was an error
int transpose(matrix_fp32 *m, matrix_fp32 **result)
{
    if (m == NULL || m->data == NULL)
        return 1;
    size_t w = m->width;
    size_t h = m->height;
    float *d = NULL;
    if (create_matrix_fp32(w, h, d, result)) return 1;
    int i, j;
    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width; j++) {
            (*result)->data[j*(*result)->width+i] = m->data[i*m->width+j];
        }
    }
    return 0;
}

// TODO: create an OpenCL version of this function, using a different algorithm
// TODO: take a pointer (int*) where we should store an error value
float determinant(matrix_fp32 *m)
{
    int i, j, k, neg;
    float sum;
    if (m->width != m->height) return INFINITY;
    if (m->width == 1) return    m->data[0];
    if (m->width == 2) return   (m->data[0] * m->data[1*m->width+1]) -
                                (m->data[1] * m->data[1*m->width]);
    sum = 0;
    neg = 1;
    for (i = 0; i < m->width; i++) {
        // Make a matrix out of the submatrix.
        size_t w = m->width - 1;
        size_t h = m->height - 1;
        matrix_fp32 *sm = NULL;
        if (create_matrix_fp32(w, h, NULL, &sm))
        {
            fprintf(stderr, "Error allocating sub-matrix for determinant calculation\n");
            if (sm) free(sm);
            return INFINITY;
        }
        sm->width = m->width - 1;
        sm->height = m->width - 1;
        // Fill in the data.
        for (j = 0; j < sm->width; j++) /* The y position in the submatrix. */{
            for (k = 0; k < sm->width; k++) /* The x position in the submatrix. */{
                int a = ((k == i) || (i+1 == k))? k+1: k; // The x position in the matrix to copy from.
                sm->data[j*sm->width+k] = m->data[(j+1)*m->width+a];
            }
        }
        float smdet = determinant(sm);
        free(sm->data);
        free(sm);
        sum += neg * m->data[i] * smdet;
        neg *= -1;
    }
    return sum;
}

// Multiply two matrices and put the result into a given pointer
// Arguments:
// a - The first matrix to multiply
// b - The second matrix to multiply
// result - A pointer that is modified to point to the matrix_fp32 struct that
//          was allocated for the result
// Return value:
// - 0 if there was no error
// - 1 if there was an error
int mat_fp32_multiply(matrix_fp32 *a, matrix_fp32 *b, matrix_fp32 **result)
{
    // Declare the OpenCL variables.
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem arr1_buf = NULL, arr2_buf = NULL, arr_result_buf = NULL, shared_dim_buf = NULL, width_buf = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    // Declare and allocate memory for the result matrix.
    if (a->width != b->height) return 1; // Can't multiply these matrices.
    int new_width = b->width;
    int new_height = a->height;
    float *new_data = (float *)malloc(sizeof(float) * new_width * new_height);
    if (new_data == NULL) return 1;
    if (create_matrix_fp32(new_height, new_width, new_data, result))
    {
        fprintf(stderr, "Error creating result matrix in matrix multiplication"
                " function\n");
        free(new_data);
        exit(EXIT_FAILURE);
    }

    // Get references to the data arrays of the matrices and store their size.
    float *arr1 = a->data;
    float *arr2 = b->data;
    int arr1_len = a->width * a->height;
    int arr2_len = b->width * b->height;
    int shared_dim = a->width;

    size_t worksizes[] = {new_width * new_height}; // The number of work items.

    // Define the source code for the kernel.
    const char matrix_multiply_cl[] = "                                               \n\
        __kernel void matrix_multiply(                                                \n\
                                      __constant float *array1,                       \n\
                                      __constant float *array2,                       \n\
                                      __global float *output,                         \n\
                                               const int shared_dim,                  \n\
                                               const int width                        \n\
                                      )                                               \n\
        {                                                                             \n\
            const size_t idx = get_global_id(0);                                      \n\
            const uint x = idx % width;                                               \n\
            const uint y = idx / width;                                               \n\
            uint i;                                                                   \n\
            output[y * width + x] = 0;                                                \n\
            for (i = 0; i < shared_dim; i++) {                                        \n\
                output[idx] += array1[y * shared_dim + i] * array2[i * width + x];    \n\
            }                                                                         \n\
        }                                                                             \n\
    ";

    const char *src = matrix_multiply_cl;
    size_t src_size = (size_t)strlen(matrix_multiply_cl);

    // Get platform and device info.
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create openCL context.
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create command queue.
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffer.
    arr1_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, arr1_len * sizeof(float), NULL, &ret);
    arr2_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, arr2_len * sizeof(float), NULL, &ret);
    arr_result_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, worksizes[0] * sizeof(float), NULL, &ret);
    shared_dim_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    width_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);

    // Create kernel program from the source.
    program = clCreateProgramWithSource(context, 1, &src, &src_size, &ret);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }

    // Build kernel program.
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }

    // Create OpenCL kernel.
    kernel = clCreateKernel(program, "matrix_multiply", &ret);

    // Set OpenCL kernel parameters.
    ret = clSetKernelArg(kernel, 0, sizeof(arr1_buf), (void *)&arr1_buf);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }
    ret = clSetKernelArg(kernel, 1, sizeof(arr2_buf), (void *)&arr2_buf);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }
    ret = clSetKernelArg(kernel, 2, sizeof(arr_result_buf), (void *)&arr_result_buf);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }
    ret = clSetKernelArg(kernel, 3, sizeof(shared_dim), (void *)&shared_dim);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }
    ret = clSetKernelArg(kernel, 4, sizeof(new_width), (void *)&new_width);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }

    // Copy data to the memory buffer.
    ret = clEnqueueWriteBuffer(command_queue, arr1_buf, CL_TRUE, 0, arr1_len * sizeof(float), arr1, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, arr2_buf, CL_TRUE, 0, arr2_len * sizeof(float), arr2, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, shared_dim_buf, CL_TRUE, 0, sizeof(int), &shared_dim, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, width_buf, CL_TRUE, 0, sizeof(int), &new_width, 0, NULL, NULL);

    // Execute OpenCL kernel.
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, worksizes, worksizes, 0, NULL, NULL);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }

    // Copy results from the memory buffer.
    ret = clEnqueueReadBuffer(command_queue, arr_result_buf, CL_TRUE, 0,
            new_height * new_width * sizeof(float), (*result)->data, 0, NULL, NULL);
    if (ret)
    {
        printf("ERROR: Line %d (%d)\n", __LINE__, ret);
        exit(EXIT_FAILURE);
    }

    clFinish(command_queue);

    return 0;
}
