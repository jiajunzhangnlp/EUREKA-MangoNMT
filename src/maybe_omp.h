#ifndef MAYBE_OMP
#define MAYBE_OMP

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num(x) 0
  #define omp_set_num_threads(n)
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif

#endif
