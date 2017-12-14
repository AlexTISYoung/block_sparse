# block_sparse
block_sparse is a python library for performing computations with 
matrices with a (hierarchical) block-sparsity structure

# Main features:

block_sparse class: class of matrices with block sparsity structure

symmetric_block_sparse: class of symmetric matrices with block sparsity structure

Each block_sparse matrix is comprised of blocks that are either zero or a block_sparse matrix, 
or (if square) a symmetric_block_sparse matrix, or a 2D numpy array

Operations supported: matrix addition, matrix multiplication, matrix (Frobenius) norm,
quadratic form

One can also convert a block_sparse or symmetric_block_sparse matrix into a 2D numpy array

# Installation
    
# Documentation

Documentation is available at http://block-sparse.readthedocs.io/en/latest/
    
# Running tests

The script test.py performs tests for all operations for both the block_sparse
and symmetric_block_sparse classes. The number of random matrices tried 
and the numerical precision of the test can be modified at the top of the script. 