#' getDeviceProps
#' 
#' The \code{getDeviceProps} function returns some properties of the CUDA
#' device specified by the \code{id} argument.
#'
#' @param id ID of the CUDA device for which properties should be returned.
#' Defaults to \code{0} for the first device.
#'
#' @export
getDeviceProps = function(id = 0) {
  
  rst <- .getDeviceProps(id = id)
  
  txt <- c("Found CUDA capable device.\n",
           "Device ID:\t\t%s\n",
           "Device name:\t\t%s\n",
           "Integraded device:\t%s\n",
           "Compute capability:\t%s\n")
  
  msg <- sprintf(paste(txt, collapse = ""),
                 rst$id,
                 rst$deviceName,
                 as.logical(rst$integr),
                 sprintf("%s.%s", rst$mjr, rst$mnr))
  cat(msg)
}

#' cuBLASdgemm
#' 
#' The \code{cuBLASdgemm} function multiplies two matrices on the GPU.
#'
#' @param A Left matrix
#' @param B Right matrix
#'
#' @return The result of \code{A \%*\% B}.
#' @export
cuBLASdgemm = function(A, B) {
  C <- .cuBLASdgemm(A, B)
  return(C)
}

#' cuBLASdgemm_ABAt
#' 
#' The \code{cuBLASdgemm_ABAt} function computes ABAt on the GPU.
#'
#' @param A Left and right (transposed) matrix
#' @param B Right matrix
#'
#' @return The result of \code{A \%*\% B \%*\% t(A)}.
#' @export
cuBLASdgemm_ABAt = function(A, B) {
  C <- .cuBLASdgemm_ABAt(A, B)
  return(C)
}

#' sgemm_naive
#' 
#' The \code{sgemm_naive} function multiplies two float32 matrices on the GPU using
#' a naive kernel.
#'
#' @param A Left matrix
#' @param B Right matrix
#'
#' @return The result of \code{A \%*\% B}.
#' @export
sgemm_naive = function(A, B) {
  C <- .sgemm_naive(t(A), t(B))
  return(t(C))
}

#' cuBLASsgemm
#' 
#' The \code{cuBLASsgemm} function multiplies two matrices on the GPU (float32).
#'
#' @param A Left matrix
#' @param B Right matrix
#'
#' @return The result of \code{A \%*\% B}.
#' @export
cuBLASsgemm = function(A, B) {
  C <- .cuBLASsgemm(A, B)
  return(C)
}


