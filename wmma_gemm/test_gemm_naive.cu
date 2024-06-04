#include <mma.h>

#include <cmath>
#include <cstdio>
#define DEBUG
using namespace nvcuda;

constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;
constexpr int WARP_SIZE = 32;

__device__ __host__ __forceinline__ int div_ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void gemm_wmma(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> A_frag;
        // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::col_major> B_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        // wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K * N + warp_col, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(double *A, double *B, double *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
    gemm_wmma<<<grid, block>>>(A, B, C, M, N, K);
}

// c = a_{M \times K} * b_{K \times N}
void gemm_cpu(double *a, double *b, double *c, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

double compare_res(double *h_res, double *d_res, int M, int N) {
    double diff = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            diff += fabs(h_res[i * N + j] - d_res[i * N + j]);
        }

        printf("h: ");
        for (int j = 0; j < N; j++) {
            printf("%.1lf ", h_res[i * N + j]);
        }
        printf(" || d: ");
        for (int j = 0; j < N; j++) {
            printf("%.1lf ", d_res[i * N + j]);
        }
        printf("\n");
    }
    return diff;
}

void init_matrix(double *matrix, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix[i] = i;
    }
}
void init_matrix_ones(double *matrix, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix[i] = 1;
    }
}

int main(int argc, char **argv) {
    double *h_a;
    double *h_b;
    double *h_c;
    double *h_d_res;

    double *d_a;
    double *d_b;
    double *d_c;

    const int M = 8;
    const int N = 8;
    const int K = 4;

    h_a = (double *)malloc(M * K * sizeof(double));
    h_b = (double *)malloc(K * N * sizeof(double));
    h_c = (double *)malloc(M * N * sizeof(double));
    h_d_res = (double *)malloc(M * N * sizeof(double));

    cudaMalloc(&d_a, M * K * sizeof(double));
    cudaMalloc(&d_b, K * N * sizeof(double));
    cudaMalloc(&d_c, M * N * sizeof(double));
    // init matrix
    init_matrix_ones(h_a, M, K);
    init_matrix(h_b, K, N);

    cudaMemcpy(d_a, h_a, M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(double), cudaMemcpyHostToDevice);
    wmmaNaive(d_a, d_b, d_c, M, N, K);
    gemm_cpu(h_a, h_b, h_c, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d_res, d_c, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    double diff = compare_res(h_c, h_d_res, M, N);
    printf("diff: %f\n", diff);

    // free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}