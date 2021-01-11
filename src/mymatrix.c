#include <stdio.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include "mymatrix.h"

#include "matrix_kernels.h"

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
// - MATRIX_SUCCESS if there was no error
// - ALLOCATION_FAILURE if there was an error
int create_matrix_fp32(size_t h, size_t w, float *d, matrix_fp32 **result)
{
    if (*result == NULL)
        *result = (matrix_fp32 *)malloc(sizeof(matrix_fp32));
    if (*result == NULL) return ALLOCATION_FAILURE;
    if (d == NULL)
        d = (float *)malloc(sizeof(float)*w*h);
    if (d == NULL) return ALLOCATION_FAILURE;
    (*result)->height = h;
    (*result)->width = w;
    (*result)->data = d;
    return MATRIX_SUCCESS;
}

// Transpose a matrix into a given pointer location.
// Arguments:
// m - matrix to greate the transpose of
// result - A pointer that is modified to point to the matrix_fp32 struct that
//          was allocated for the transpose
// Return value:
// - 0 if there was no error
// - <ERROR_CODE> if there was an error:
//      - INVALID_MATRIX if the matrix or its data pointer are NULL
//      - ALLOCATION_FAILURE if creating the result matrix fails
int transpose(matrix_fp32 *m, matrix_fp32 **result)
{
    int ret;
    if (m == NULL || m->data == NULL)
        return INVALID_MATRIX;
    size_t w = m->width;
    size_t h = m->height;
    float *d = NULL;
    if ((ret = create_matrix_fp32(w, h, d, result)) != MATRIX_SUCCESS) return ret;
    int i, j;
    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width; j++) {
            (*result)->data[j*(*result)->width+i] = m->data[i*m->width+j];
        }
    }
    return MATRIX_SUCCESS;
}

// TODO: create an OpenCL version of this function, using a different algorithm
float determinant(matrix_fp32 *m, int *err)
{
    int i, j, k, neg;
    int ret;
    float sum;
    if (m->width != m->height)
    {
        if (err != NULL) *err = INVALID_SHAPES; // Non-square matrices do not have determinants.
        return INFINITY;
    }
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
        if ((ret = create_matrix_fp32(w, h, NULL, &sm)) != MATRIX_SUCCESS)
        {
            fprintf(stderr, "Error allocating sub-matrix for determinant calculation\n");
            if (sm) free(sm);
            if (err != NULL) *err = ret;
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
        float smdet = determinant(sm, err);
        free(sm->data);
        free(sm);
        sum += neg * m->data[i] * smdet;
        neg *= -1;
    }
    if (err != NULL) *err = MATRIX_SUCCESS;
    return sum;
}

// Multiply two matrices and put the result into a given pointer
// Arguments:
// a - The first matrix to multiply
// b - The second matrix to multiply
// result - A pointer that is modified to point to the matrix_fp32 struct that
//          was allocated for the result
// Return value:
// - MATRIX_SUCCESS if there was no error
// - <ERROR_CODE> if there was an error:
//      - ALLOCATION_FAILURE if allocating the data for the result matrix fails
//      - INVALID_SHAPES if the matrices are the wrong shape for multiplication
//      - OPENCL_ERROR if there was an error with the OpenCL setup/execution
int mat_fp32_multiply(matrix_fp32 *a, matrix_fp32 *b, matrix_fp32 **result)
{
    int err; // Error code when creating result matrix
    char allocd_mat = 0;
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
    if (a->width != b->height) return INVALID_SHAPES; // Can't multiply these matrices.
    int new_width = b->width;
    int new_height = a->height;
    if (*result == NULL)
    {
        float *new_data = (float *)malloc(sizeof(float) * new_width * new_height);
        if (new_data == NULL) return ALLOCATION_FAILURE;
        if ((err = create_matrix_fp32(new_height, new_width, new_data, result)) != MATRIX_SUCCESS)
        {
            fprintf(stderr, "Error creating result matrix in matrix multiplication"
                    " function\n");
            free(new_data);
            return err;
        }
        allocd_mat = 1;
    }

    // Get references to the data arrays of the matrices and store their size.
    float *arr1 = a->data;
    float *arr2 = b->data;
    int arr1_len = a->width * a->height;
    int arr2_len = b->width * b->height;
    int shared_dim = a->width;

    size_t worksizes[] = {new_width * new_height}; // The number of work items.

    size_t src_size = (size_t)strlen(matrix_multiply_cl);

    // Get platform and device info.
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    if (ret) goto CLEANUP_OCL_ERR;

    // Create openCL context.
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret) goto CLEANUP_OCL_ERR;

    // Create command queue.
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret) goto CLEANUP_OCL_ERR;

    // Create memory buffer.
    arr1_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, arr1_len * sizeof(float), NULL, &ret);
    if (ret) goto CLEANUP_OCL_ERR;
    arr2_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, arr2_len * sizeof(float), NULL, &ret);
    if (ret) goto CLEANUP_OCL_ERR;
    arr_result_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, worksizes[0] * sizeof(float), NULL, &ret);
    if (ret) goto CLEANUP_OCL_ERR;
    shared_dim_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    if (ret) goto CLEANUP_OCL_ERR;
    width_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    if (ret) goto CLEANUP_OCL_ERR;

    // Create kernel program from the source.
    program = clCreateProgramWithSource(context, 1, &matrix_multiply_cl, &src_size, &ret);
    if (ret) goto CLEANUP_OCL_ERR;

    // Build kernel program.
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;

    // Create OpenCL kernel.
    kernel = clCreateKernel(program, "matrix_multiply", &ret);
    if (ret) goto CLEANUP_OCL_ERR;

    // Set OpenCL kernel parameters.
    ret = clSetKernelArg(kernel, 0, sizeof(arr1_buf), (void *)&arr1_buf);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clSetKernelArg(kernel, 1, sizeof(arr2_buf), (void *)&arr2_buf);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clSetKernelArg(kernel, 2, sizeof(arr_result_buf), (void *)&arr_result_buf);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clSetKernelArg(kernel, 3, sizeof(shared_dim), (void *)&shared_dim);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clSetKernelArg(kernel, 4, sizeof(new_width), (void *)&new_width);
    if (ret) goto CLEANUP_OCL_ERR;

    // Copy data to the memory buffer.
    ret = clEnqueueWriteBuffer(command_queue, arr1_buf, CL_TRUE, 0, arr1_len * sizeof(float), arr1, 0, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clEnqueueWriteBuffer(command_queue, arr2_buf, CL_TRUE, 0, arr2_len * sizeof(float), arr2, 0, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clEnqueueWriteBuffer(command_queue, shared_dim_buf, CL_TRUE, 0, sizeof(int), &shared_dim, 0, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;
    ret = clEnqueueWriteBuffer(command_queue, width_buf, CL_TRUE, 0, sizeof(int), &new_width, 0, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;

    // Execute OpenCL kernel.
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, worksizes, worksizes, 0, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;

    // Copy results from the memory buffer.
    ret = clEnqueueReadBuffer(command_queue, arr_result_buf, CL_TRUE, 0,
            new_height * new_width * sizeof(float), (*result)->data, 0, NULL, NULL);
    if (ret) goto CLEANUP_OCL_ERR;

    clFinish(command_queue);

    return 0;

CLEANUP_OCL_ERR:
    if (allocd_mat)
    {
        free((*result)->data);
        free(*result);
    }
    return OPENCL_ERROR;
}
