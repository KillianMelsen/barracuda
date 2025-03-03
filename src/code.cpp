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

// dgemm function
// C = alpha * op(A) %*% op(B) + beta * C
//[[Rcpp::export(.dgemm)]]
arma::mat dgemm(const arma::mat& A, const arma::mat& B) {
  
  // Matrix dimensions (left, inner, right)
  int L = A.n_rows;
  int I = A.n_cols;
  int R = B.n_rows;
  
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
  arma::mat C(L, R);
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


