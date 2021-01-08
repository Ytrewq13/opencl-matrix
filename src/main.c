#include <stdio.h>
#include <stdlib.h>

#include <strings.h>

#define CL_TARGET_OPENCL_VERSION 120

#include "mymatrix.h"

#include <CL/cl.h>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define ARR_SIZE (24)

#define ARR_SRC {0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f}

/* TODO:
 * - Refactor
 *   - Remove boilerplate from program logic (move program logic into internal
 *     function of "mymatrix.h" and just do: b.plate, call function, b.plate in
 *     the main function).
 * - TEST!!!!!
 *   - I have no idea if the determinant actually works.
 *   - Test matrix multiplication for non-square matrices.
 *   - Test with some random numbers (against a real implementation in -lmath?)
 * - COMMENT
 *   - I wasted too much time understanding the spaghetti I found here earlier
 *     today. I have so far left the code in a more coherent state than I found
 *     it in but it is still not good.
 */

void matrix_test() {
    int dim = 4;
    float matrix_data[] = ARR_SRC;
    matrix_fp32 *mat = NULL;
    if (create_matrix_fp32(dim, dim, matrix_data, &mat))
    {
        fprintf(stderr, "Error creating matrix in matrix test function\n");
        exit(EXIT_FAILURE);
    }
    print_matrix(mat);
    printf("%f\n\n", determinant(mat));

    float matrix_data2[] = ARR_SRC;
    matrix_fp32 *mat2 = NULL;
    if (create_matrix_fp32(dim, dim, matrix_data2, &mat2))
    {
        fprintf(stderr, "Error creating matrix in matrix test function\n");
        free(mat);
        exit(EXIT_FAILURE);
    }
    print_matrix(mat2);
    printf("%f\n\n", determinant(mat2));

    matrix_fp32 *mat_mult_result = NULL;
    if (mat_fp32_multiply(mat, mat2, &mat_mult_result))
    {
        fprintf(stderr, "Failed to multiply matrices\n");
        free(mat);
        free(mat2);
        exit(EXIT_FAILURE);
    }

    printf("Width: %i, Height: %i\n", mat_mult_result->width,
            mat_mult_result->height);

    print_matrix(mat_mult_result);
    printf("%f\n", determinant(mat_mult_result));

    free(mat);
    free(mat2);
    free(mat_mult_result);
}

int main(int argc, char **argv) {
    matrix_test();
    return 0;
}
