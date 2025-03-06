// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace Rcpp;



// Device properties function
//[[Rcpp::export(.getDeviceProps)]]
List getDeviceProps(const int& id) {
  
  // Create CUDA device properties variable
  cudaDeviceProp prop;
  
  // Query GPU properties
  cudaGetDeviceProperties(&prop, id);
  
  // Storing and returning results
  char deviceName[256];
  strcpy(deviceName, prop.name);
  return List::create(Named("id") = id,
                      Named("deviceName") = deviceName,
                      Named("integr") = prop.integrated,
                      Named("mjr") = prop.major,
                      Named("mnr") = prop.minor);
}



// cuBLASdgemm function
// C = alpha * op(A) %*% op(B) + beta * C
//[[Rcpp::export(.cuBLASdgemm)]]
arma::dmat cuBLASdgemm(const arma::dmat& A, const arma::dmat& B) {
  
  // Matrix dimensions (left, inner, right)
  int L = A.n_rows;
  int I = A.n_cols;
  int R = B.n_cols;
  
  // Create CUDA error variable, cuBLAS handle, and CUDA stream
  cudaError_t err;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  
  // We want normal matrix multiplication
  const double alpha = 1.0;
  const double beta = 0.0;
  
  // Creating device matrices
  double* d_A = nullptr;
  double* d_B = nullptr;
  double* d_C = nullptr;
  // Rcout << "Device matrices created...\n";
  
  // op(A) and op(B), we want normal matrix multiplication, so no transposing
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  
  // Create cuBLAS handle and bind stream
  cublasCreate(&cublasH);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);
  // Rcout << "cuBLAS handle and CUDA stream created and bound...\n";
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  
  // Allocating device memory and copying A and B to the device
  size_t size = (L) * (I) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_A), size);
  cudaMemcpyAsync(d_A, A.memptr(), size, cudaMemcpyHostToDevice, stream);
  // Rcout << "Memory for A allocated on device...\n";
  
  size = (I) * (R) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_B), size);
  cudaMemcpyAsync(d_B, B.memptr(), size, cudaMemcpyHostToDevice, stream);
  // Rcout << "Memory for B allocated on device...\n";
  
  size = (L) * (R) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_C), size);
  // Rcout << "Memory for C allocated on device...\n";
  
  // Error checking before executing device code
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  
  // Computing
  // Rcout << "Executing device code...\n";
  cublasDgemm(cublasH, transa, transb, L, R, I, &alpha, d_A, L, d_B, I, &beta, d_C, L);
  
  // Error checking after executing device code
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "device code successfully executed...\n";
  
  // Copying from the device
  arma::dmat C(L, R);
  cudaMemcpyAsync(C.memptr(), d_C, size, cudaMemcpyDeviceToHost, stream);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "C copied from device to host...\n";
  
  // Synchronizing, freeing memory, etc.
  cudaStreamSynchronize(stream);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(cublasH);
  cudaStreamDestroy(stream);
  
  // Final error check
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    cudaDeviceReset();
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "Device memory freed, cuBLAS handle and CUDA stream destroyed...\n";
  
  cudaDeviceReset();
  // Rcout << "Device resetted, returning C...\n";
  return C;
}



// cuBLASsgemm function
// C = alpha * op(A) %*% op(B) + beta * C
//[[Rcpp::export(.cuBLASsgemm)]]
arma::fmat cuBLASsgemm(const arma::fmat& A, const arma::fmat& B) {
  
  // Matrix dimensions (left, inner, right)
  int L = A.n_rows;
  int I = A.n_cols;
  int R = B.n_cols;
  
  // Create CUDA error variable, cuBLAS handle, and CUDA stream
  cudaError_t err;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  
  // We want normal matrix multiplication
  const float alpha = 1.0;
  const float beta = 0.0;
  
  // Creating device matrices
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  // Rcout << "Device matrices created...\n";
  
  // op(A) and op(B), we want normal matrix multiplication, so no transposing
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  
  // Create cuBLAS handle and bind stream
  cublasCreate(&cublasH);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);
  // Rcout << "cuBLAS handle and CUDA stream created and bound...\n";
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  
  // Allocating device memory and copying A and B to the device
  size_t size = (L) * (I) * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_A), size);
  cudaMemcpyAsync(d_A, A.memptr(), size, cudaMemcpyHostToDevice, stream);
  // Rcout << "Memory for A allocated on device...\n";
  
  size = (I) * (R) * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_B), size);
  cudaMemcpyAsync(d_B, B.memptr(), size, cudaMemcpyHostToDevice, stream);
  // Rcout << "Memory for B allocated on device...\n";
  
  size = (L) * (R) * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_C), size);
  // Rcout << "Memory for C allocated on device...\n";
  
  // Error checking before executing device code
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  
  // Computing
  // Rcout << "Executing device code...\n";
  cublasSgemm(cublasH, transa, transb, L, R, I, &alpha, d_A, L, d_B, I, &beta, d_C, L);
  
  // Error checking after executing device code
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "device code successfully executed...\n";
  
  // Copying from the device
  arma::fmat C(L, R);
  cudaMemcpyAsync(C.memptr(), d_C, size, cudaMemcpyDeviceToHost, stream);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "C copied from device to host...\n";
  
  // Synchronizing, freeing memory, etc.
  cudaStreamSynchronize(stream);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(cublasH);
  cudaStreamDestroy(stream);
  
  // Final error check
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    cudaDeviceReset();
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "Device memory freed, cuBLAS handle and CUDA stream destroyed...\n";
  
  cudaDeviceReset();
  // Rcout << "Device resetted, returning C...\n";
  return C;
}



// cuBLASdgemm_ABAt function
// C = A %*% B
// D = C %*% t(A)
//[[Rcpp::export(.cuBLASdgemm_ABAt)]]
arma::dmat cuBLASdgemm_ABAt(const arma::dmat& A, const arma::dmat& B) {
  
  // Matrix dimensions (left, inner, right)
  int L = A.n_rows;
  int I = A.n_cols;
  int R = A.n_rows;
  
  // Create CUDA error variable, cuBLAS handle, and CUDA stream
  cudaError_t err;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  
  // We want normal matrix multiplication
  const double alpha = 1.0;
  const double beta = 0.0;
  
  // Creating device matrices
  double* d_A = nullptr; // A
  double* d_B = nullptr; // B
  double* d_C = nullptr; // AB
  double* d_D = nullptr; // CAt
  // Rcout << "Device matrices created...\n";
  
  // op(A) and op(B), we want normal matrix multiplication, so no transposing
  cublasOperation_t transa1 = CUBLAS_OP_N;
  cublasOperation_t transb1 = CUBLAS_OP_N;
  cublasOperation_t transa2 = CUBLAS_OP_N;
  cublasOperation_t transb2 = CUBLAS_OP_T;
  
  // Create cuBLAS handle and bind stream
  cublasCreate(&cublasH);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);
  // Rcout << "cuBLAS handle and CUDA stream created and bound...\n";
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  
  // Allocating device memory and copying A and B to the device
  size_t size = (L) * (I) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_A), size);
  cudaMemcpyAsync(d_A, A.memptr(), size, cudaMemcpyHostToDevice, stream);
  // Rcout << "Memory for A allocated on device...\n";
  
  size = (I) * (I) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_B), size);
  cudaMemcpyAsync(d_B, B.memptr(), size, cudaMemcpyHostToDevice, stream);
  // Rcout << "Memory for B allocated on device...\n";
  
  size = (L) * (I) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_C), size);
  // Rcout << "Memory for C allocated on device...\n";
  
  size = (L) * (R) * sizeof(double);
  cudaMalloc(reinterpret_cast<void**>(&d_D), size);
  // Rcout << "Memory for D allocated on device...\n";
  
  // Error checking before executing device code
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  
  // Computing C = AB
  // Rcout << "Executing device code 1...\n";
  cublasDgemm(cublasH, transa1, transb1, L, I, I, &alpha, d_A, L, d_B, I, &beta, d_C, L);
  
  // Computing D = CAt
  // Rcout << "Executing device code 2...\n";
  cublasDgemm(cublasH, transa2, transb2, L, R, I, &alpha, d_C, L, d_A, L, &beta, d_D, L);
  
  // Error checking after executing device code
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "device code successfully executed...\n";
  
  // Copying from the device
  arma::dmat D(L, R);
  cudaMemcpyAsync(D.memptr(), d_D, size, cudaMemcpyDeviceToHost, stream);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "C copied from device to host...\n";
  
  // Synchronizing, freeing memory, etc.
  cudaStreamSynchronize(stream);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);
  cublasDestroy(cublasH);
  cudaStreamDestroy(stream);
  
  // Final error check
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    Rcerr << "CUDA ERROR: " << err << "\n";
    cudaDeviceReset();
    stop("CUDA ERROR: %i", err);
  }
  // Rcout << "Device memory freed, cuBLAS handle and CUDA stream destroyed...\n";
  
  cudaDeviceReset();
  // Rcout << "Device resetted, returning C...\n";
  return D;
}



// DIY sgemm kernel (reading/saving from storage in row-major format)
// https://siboehm.com/articles/22/CUDA-MMM
__global__ void sgemm_naive_kernel(int L, int R, int I, float alpha, const float* A,
                                   const float* B, float beta, float* C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x; // column
  const uint y = blockIdx.y * blockDim.y + threadIdx.y; // row

  // `if` condition is necessary for when L or R aren't multiples of 32.
  if (y < L && x < R) {
    float tmp = 0.0;
    for (int i = 0; i < I; ++i) {
      tmp += A[y * I + i] * B[i * R + x];
    }
    // C = α * (A %*% B) + β * C
    // For a dim(A)=2x2 and dim(B) = 2x2, the topright element (x = 1, y = 0)
    // would end up in the second spot of the linear memory array. That would
    // be correct for a row-major storage system, but we need to transpose C
    // (in the R wrapper) to make it correct for R's/RcppArmadillo's col-major
    // format.
    C[y * R + x] = alpha * tmp + beta * C[y * R + x];
  }
}

// sgemm_naive function
// C = A %*% B
//[[Rcpp::export(.sgemm_naive)]]
arma::fmat sgemm_naive(const arma::fmat& A, const arma::fmat& B) {

  // Matrix dimensions (left, inner, right)
  // Matrices have been transposed in R because R/RcppArmadillo use col-major
  // format while CUDA typically uses row-major format.
  int L = A.n_cols;
  int R = B.n_rows;
  int I = A.n_rows;

  // Create as many blocks as necessary to map all of C
  dim3 dimGrid(ceil(L / (double) 32), ceil(R / (double) 32), 1);

  // 32 * 32 = 1024 thread per block
  dim3 dimBlock(32, 32, 1);

  // We want normal matrix multiplication
  const float alpha = 1.0;
  const float beta = 0.0;

  // Creating device matrices
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;

  // Allocating device memory and copying A and B to the device
  size_t size = (L) * (I) * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_A), size);
  cudaMemcpyAsync(d_A, A.memptr(), size, cudaMemcpyHostToDevice);
  // Rcout << "Memory for A allocated on device...\n";

  size = (I) * (R) * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_B), size);
  cudaMemcpyAsync(d_B, B.memptr(), size, cudaMemcpyHostToDevice);
  // Rcout << "Memory for B allocated on device...\n";

  size = (L) * (R) * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_C), size);
  // Rcout << "Memory for C allocated on device...\n";

  // Calling the kernel
  sgemm_naive_kernel<<<dimGrid, dimBlock>>>(L, R, I, alpha, d_A, d_B, beta, d_C);

  // Copying from the device
  arma::fmat C(L, R);
  cudaMemcpyAsync(C.memptr(), d_C, size, cudaMemcpyDeviceToHost);

  // Synchronizing, freeing memory, etc.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  cudaDeviceReset();
  return C;
}


