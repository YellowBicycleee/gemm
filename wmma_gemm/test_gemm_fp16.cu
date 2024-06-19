#include <mma.h>

#include <cmath>
#include <cstdio>
#define DEBUG
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

__device__ __host__ __forceinline__ int div_ceil(int a, int b) { return (a + b - 1) / b; }
__global__ void gemm_wmma(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                          size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float,
        // wmma::col_major> B_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        // wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K * N + warp_col, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

__global__ void gemm_wmma_shared(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                 size_t N, size_t K) {
    __shared__ half A_shared[WMMA_M * WMMA_K];
    __shared__ half B_shared[WMMA_K * WMMA_N];
    __shared__ half C_shared[WMMA_M * WMMA_N];

    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    int m;
    int n;
    int global_m;
    int global_n;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        // load a
        // wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        for (int j = threadIdx.x; j < WMMA_M * WMMA_K; j += WARP_SIZE) {
            m = j / WMMA_K;  // smem index
            n = j % WMMA_K;  // smem index
            global_m = warp_row + m;
            global_n = i * WMMA_K + n;
            // global_index  i_A = warp_row + m, j_A = i * WMMA_K + n
            if (global_m < M && global_n < K) {
                A_shared[j] = A[global_m * K + global_n];
            } else {
                A_shared[j] = 0;  // Zero padding
            }
        }
        // load b
        for (int j = threadIdx.x; j < WMMA_K * WMMA_N; j += WARP_SIZE) {
            m = j / WMMA_N;  // smem   index
            n = j % WMMA_N;  // smem   index
            global_m = i * WMMA_K + m;
            global_n = warp_col + n;
            // global_index, i_B = i * WMMA_K + m, j_B = warp_col + n
            if (global_m < K && global_n < N) {
                B_shared[j] = B[global_m * N + global_n];
            } else {
                B_shared[j] = 0;  // Zero padding
            }
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float,
        // wmma::col_major> B_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

        // wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        // wmma::load_matrix_sync(B_frag, B + i * WMMA_K * N + warp_col, N);

        wmma::load_matrix_sync(A_frag, A_shared, WMMA_K);
        wmma::load_matrix_sync(B_frag, B_shared, WMMA_N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
    // store to smem c
    // wmma::store_matrix_sync(C_shared, C_frag, N, wmma::mem_row_major);
    // for (int j = threadIdx.x; j < WMMA_M * WMMA_N; j += WARP_SIZE) {
    //     m = j / WMMA_N;
    //     n = j % WMMA_N;
    //     global_m = warp_row + m;
    //     global_n = warp_col + n;
    //     // global, i_C = warp_row + m; j_C = warp_col + n
    //     // C[(warp_row + m) * N + (warp_col + n)] = C_shared[j];
    //     if (global_m < M && global_n < N) {
    //         C[(warp_row + m) * N + (warp_col + n)] = C_shared[j];
    //     }
    // }
    // __syncwarp();
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
    // gemm_wmma<<<grid, block>>>(A, B, C, M, N, K);
    gemm_wmma_shared<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    // gemm_wmma_shared<<<grid, block>>>(A, B, C, M, N, K);
}

// c = a_{M \times K} * b_{K \times N}
void gemm_cpu(half *a, half *b, half *c, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(a[i * K + k]) * __half2float(b[k * N + j]);
            }
            c[i * N + j] = __float2half(sum);
            // printf("sum = %e\n", __half2float(c[i * N + j]));
        }
    }
}

float compare_res(half *h_res, half *d_res, int M, int N) {
    float diff = 0;
    const int NEW_M = max(M, 8);
    const int NEW_N = max(N, 16);

    for (int i = 0; i < NEW_M; ++i) {
        for (int j = 0; j < NEW_N; ++j) {
            diff += fabs(__half2float(h_res[i * N + j]) - __half2float(d_res[i * N + j]));
        }
        // printf("diff = %e\n", diff);

        printf("h: ");
        for (int j = 0; j < N; ++j) {
            printf("%.1lf ", __half2float(h_res[i * N + j]));
        }
        printf(" || d: ");
        for (int j = 0; j < N; ++j) {
            printf("%.1lf ", __half2float(d_res[i * N + j]));
        }
        printf("\n");
    }
    return diff;
}

void init_matrix(half *matrix, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        matrix[i] = i;
    }
}

void print_matrix(half *matrix, int M, int N) {
    printf("==============start==================\n");
    const int NEW_M = max(M, 8);
    const int NEW_N = max(N, 16);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4.2lf ", __half2float(matrix[i * N + j]));
        }
        printf("\n");
    }
    printf("==============end==================\n");
}

void init_matrix_ones(half *matrix, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        matrix[i] = 1;
    }
}

int main(int argc, char **argv) {
    half *h_a;
    half *h_b;
    half *h_c;
    half *h_d_res;

    half *d_a;
    half *d_b;
    half *d_c;

    const int M = 32;
    const int N = 32;
    const int K = 32;

    h_a = (half *)malloc(M * K * sizeof(half));
    h_b = (half *)malloc(K * N * sizeof(half));
    h_c = (half *)malloc(M * N * sizeof(half));
    h_d_res = (half *)malloc(M * N * sizeof(half));

    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(half));
    // init matrix
    init_matrix_ones(h_a, M, K);
    init_matrix(h_b, K, N);
    print_matrix(h_a, M, K);
    print_matrix(h_b, K, N);

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);
    wmmaNaive(d_a, d_b, d_c, M, N, K);

    gemm_cpu(h_a, h_b, h_c, M, N, K);
    print_matrix(h_c, M, N);

    cudaMemcpy(h_d_res, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    print_matrix(h_d_res, M, N);
    float diff = compare_res(h_c, h_d_res, M, N);
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