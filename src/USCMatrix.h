#ifndef USCMATRIX_H
#define USCMATRIX_H

#include <Eigen/Dense>
#include "maybe_omp.h"
#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

// USC = Uniform Sparse Columns. A USCMatrix is a sparse matrix in which
// each column has exactly k nonzero entries. This allows for a
// simpler and faster compressed representation.

// A USCMatrix can be converted into CSC format fairly easily, by
// adding a third array [0, k, 2k, ..., nk]. However, the indices will
// not be unique.

// We use:
//       dense2 = dense1^T * sparse (output bProp, input fProp)
//       dense1 = sparse * dense2^T (output computeGradient, input computeGradient)
// where:
//       sparse is vocab_size x minibatch_size
//       dense1 is vocab_size x embedding_dimension
//       dense2 is embedding_dimension x minibatch_size

template <typename Scalar, typename Index=int> // should be EIGEN_DEFAULT_DENSE_INDEX_TYPE but int is smaller
class USCMatrix
{

public:
    Matrix<Index,Dynamic,Dynamic> indexes;
    Matrix<Scalar,Dynamic,Dynamic> values;
    int m_rows;

    USCMatrix() : m_rows(0) { }

    template <typename Indexes, typename Values>
    USCMatrix(Index rows, const MatrixBase<Indexes> &indexes, const MatrixBase<Values> &values) 
    : 
      indexes(indexes), 
      values(values), 
      m_rows(rows) 
    { }

    USCMatrix(Index rows, Index nnz, Index cols) 
    : 
      indexes(Matrix<Index,Dynamic,Dynamic>(nnz, cols)), 
      values(Matrix<Scalar,Dynamic,Dynamic>(nnz, cols)),
      m_rows(rows)
    { 
        this->indexes.fill(-1); 
    }

    Index rows() const { return m_rows; }
    Index cols() const { return indexes.cols(); }

    void resize(Index rows, Index nnz, Index cols) {
        indexes.resize(nnz, cols);
        values.resize(nnz, cols);
	m_rows = rows;
    }
};

// Dense matrix - sparse matrix product
// a is presumably very wide
// Used for fProp in Input_word_embeddings class
template <typename DerivedA, typename ScalarB, typename Index, typename DerivedC>
void uscgemm(precision_type alpha, 
		 const MatrixBase<DerivedA> &a, 
	     const USCMatrix<ScalarB,Index> &b,
	     const MatrixBase<DerivedC> &c_const)
{
    UNCONST(DerivedC, c_const, c);
    eigen_assert(a.rows() == c.rows());
    eigen_assert(a.cols() == b.rows());
    eigen_assert(b.cols() == c.cols());

    #pragma omp parallel for
    for (Index k=0; k<b.cols(); k++)
        for (Index r=0; r<b.indexes.rows(); r++)
	{
	    Index j = b.indexes(r,k);
	    eigen_assert(j >= 0);
	    eigen_assert(j < a.cols());
	    c.col(k) += alpha * a.col(j) * b.values(r,k);
	}
}

// sparse matrix - dense matrix product
template <typename ScalarA, typename Index, typename DerivedB, typename DerivedC>
void uscgemm(precision_type alpha, 
	     const USCMatrix<ScalarA,Index> &a,
	     const MatrixBase<DerivedB> &b, 
	     const MatrixBase<DerivedC> &c_const)
{
    UNCONST(DerivedC, c_const, c);
    eigen_assert(a.rows() == c.rows());
    eigen_assert(a.cols() == b.rows());
    eigen_assert(b.cols() == c.cols());

    // This needs to be tuned for each system, unfortunately,
    // and seems to vary a lot. A lot.
    int i_blocks = omp_get_num_threads()*16;

    // Assume only one block in k direction.
    // We don't need to explicitly block in the j direction.
    #pragma omp parallel for
    for (Index ib=0; ib<i_blocks; ib++)
        for (Index j=0; j<a.cols(); j++)
	    for (Index r=0; r<a.indexes.rows(); r++)
	    {
	        Index i = a.indexes(r,j);
		eigen_assert(i >= 0);
		eigen_assert(i < c.rows());
		if (i % i_blocks == ib)
		    c.row(i) += alpha * a.values(r,j) * b.row(j);
	    }

    /*
    If c.cols() is really large, then theoretically it seems like we should do:

    parallel for blocks in i direction
        for blocks in j direction
            pack block of a into smaller sparse matrix
            for blocks in k direction
                for k
                    for i (sparse)
                        for j
                            c(i,k) += a(i,j) * b(j,k)

    However, the copying of blocks of a doesn't seem practical for any realistic
    sizes of c.cols().
    */
}

// Dense matrix - dense matrix product, but masked by a sparse matrix,
// that is, compute a*b only for those positions in c.indexes, and put
// them in c.values.

// a is presumably a very tall matrix. Row-major order is preferred.
// For b, column-major is preferred.

template <typename DerivedA, typename DerivedB, typename ScalarC, typename Index>
void uscgemm_masked(precision_type alpha,
		    const MatrixBase<DerivedA> &a,
		    const MatrixBase<DerivedB> &b,
		    USCMatrix<ScalarC,Index> &c)
{
	//std::cerr<<"a "<<a<<std::endl;
	//std::cerr<<"b "<<b<<std::endl;
	//cerr<<"c "<<c<<endl;
    eigen_assert(a.rows() == c.rows());
    eigen_assert(a.cols() == b.rows());
    eigen_assert(b.cols() == c.cols());

    #pragma omp parallel for
    for (Index k=0; k<b.cols(); k++)
        for (Index r=0; r<c.indexes.rows(); r++)
	{
	    Index i = c.indexes(r, k);
	    eigen_assert(i >= 0);
	    eigen_assert(i < a.rows());
	    c.values(r, k) += alpha * a.row(i) * b.col(k);
	}
}

template <typename DerivedA, typename DerivedB, typename ScalarC, typename Index>
void uscgemm_masked_nce(precision_type alpha,
		    const MatrixBase<DerivedA> &a,
		    const MatrixBase<DerivedB> &b,
		    USCMatrix<ScalarC,Index> &c)
{
    eigen_assert(a.rows() == c.rows());
    eigen_assert(a.cols() == b.rows());
    eigen_assert(b.cols() == c.cols());

    #pragma omp parallel for
    for (Index k=0; k<b.cols(); k++)
        for (Index r=0; r<c.indexes.rows(); r++)
	{
	    Index i = c.indexes(r, k);
		if (i == -1){
			continue;
		}
	    eigen_assert(i >= 0);
	    eigen_assert(i < a.rows());
	    c.values(r, k) += alpha * a.row(i) * b.col(k);
	}
}

// sparse matrix - dense vector product
template <typename ScalarA, typename Index, typename DerivedB, typename DerivedC>
void uscgemv(precision_type alpha, 
	     const USCMatrix<ScalarA,Index> &a,
	     const MatrixBase<DerivedB> &b,
	     const MatrixBase<DerivedC> &c_const)
{
    UNCONST(DerivedC, c_const, c);
    eigen_assert(a.rows() == c.rows());
    eigen_assert(a.cols() == b.rows());
    eigen_assert(b.cols() == 1 && c.cols() == 1);

    for (Index j=0; j<a.cols(); j++)
        for (Index r=0; r<a.indexes.rows(); r++)
	{
	    Index i = a.indexes(r,j);
	    eigen_assert(i >= 0);
	    eigen_assert(i < c.rows());
	    c(i) += alpha * a.values(r,j) * b(j);
	}
}

}

#endif
