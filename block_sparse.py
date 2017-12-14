import numpy as np

class block_sparse(object):
    """Define a block-sparse matrix

    Parameters
    ----------
    blocks : :class:`list`
        list of [row_block_boundaries,col_block_boundaries], where each of row_block_boundaries and col_block_boundaries is
        a 1D :class:`~numpy:numpy.array` of integers, beginning with 0, followed by the end boundaries of each block in increasing order
    nonzero : :class:`~numpy:numpy.array`
        boolean numpy array with number of rows equal to number of row blocks, and number of columns equal to number of
        col blocks. If entry [i,j] of nonzero is True, then the corresponding block is non-zero; if it is False, then the
        corresponding block is zero.
    submatrices : :class:`list`
        list of submatrices for non-zero blocks in row-major order; e.g., block (1,1), (1,2), (2,1), (2,2),...
        Each submatrix can be an :class:`~numpy:numpy.array`, a :class:`block_sparse` matrix, or a :class:`symmetric_block_sparse` matrix.
    dtype : numpy data type object
        Set the default data type for the submatrices. Default :class:`~numpy:numpy.float32`
    row_names : :class:`~numpy:numpy.array`
        numpy array with names of the row-blocks. Default :class:`None`
    col_names : :class:`~numpy:numpy.array`
        numpy array with names of the col-blocks. Default :class:`None`

    Returns
    -------
    matrix : :class:`block_sparse`
        block-sparse matrix
    """
    def __init__(self, blocks, nonzero, submatrices, dtype=np.float32, row_names=None, col_names=None):
        # Block information
        if type(blocks) == list and len(blocks) == 2:
            self.n_blocks = list()
            self.blocks = list()
            for rblocks in blocks:
                if rblocks.dtype == int and len(rblocks.shape) == 1:
                    self.blocks.append(rblocks)
                    self.n_blocks.append(rblocks.shape[0] - 1)
                else:
                    raise (ValueError('block boundaries must be 1D numpy integer array'))
            self.shape = (self.blocks[0][self.n_blocks[0]], self.blocks[1][self.n_blocks[1]])
        else:
            raise (ValueError('block boundaries must be given as a list of 1D numpy integer arrays'))
        if row_names is not None:
            if not row_names.shape[0] == self.shape[0]:
                raise (ValueError('Length of row names does not match shape of matrix'))
        if col_names is not None:
            if not col_names.shape[0] == self.shape[1]:
                raise (ValueError('Length of col names does not match shape of matrix'))
        self.row_names = row_names
        self.col_names = col_names
        # non-zero submatrix information
        if nonzero.dtype == bool and len(nonzero.shape) == 2:
            if nonzero.shape[0] == self.n_blocks[0] and nonzero.shape[1] == self.n_blocks[1]:
                self.nonzero = nonzero
                self.n_nonzero = np.sum(self.nonzero)
            else:
                raise (ValueError('Number of blocks does not match non-zero pattern'))
        else:
            raise (ValueError('non-zero pattern must be boolean array'))
        # submatrices
        if len(submatrices) == self.n_nonzero:
            # Check datatypes
            if len(submatrices) > 0:
                # Dictionary of types of submatrices
                self.types = {}
                # Mapping of nonzero to submatrices
                self.submatrix_dict = {}
                nonzero_count = 0
                for i in xrange(0, self.n_blocks[0]):
                    for j in xrange(0, self.n_blocks[1]):
                        if self.nonzero[i, j]:
                            # Check submatrices of correct dimension
                            ij_shape = (
                            self.blocks[0][i + 1] - self.blocks[0][i], self.blocks[1][j + 1] - self.blocks[1][j])
                            submatrix_type = type(submatrices[nonzero_count])
                            if submatrices[nonzero_count].shape == ij_shape and submatrix_type in [block_sparse,
                                                                                                   symmetric_block_sparse,
                                                                                                   np.ndarray]:
                                self.submatrix_dict[(i, j)] = submatrices[nonzero_count]
                                self.types[(i, j)] = submatrix_type
                                # go to next submatrix
                                nonzero_count += 1
                            else:
                                raise (ValueError('block ' + str(i) + ',' + str(j) + ' has shape ' + str(
                                    submatrices[nonzero_count].shape[0]) + ',' + str(
                                    submatrices[nonzero_count].shape[1]) + '. Expected ' + str(ij_shape[0]) + ',' + str(
                                    ij_shape[1])))
                self.submatrices = submatrices
            else:
                self.types = None
                self.submatrices = None
                self.submatrix_dict = None
            self.dtype = dtype
        else:
            raise (ValueError('Number of submatrices does not match number of non-zero blocks'))

    def get_submatrix(self, block):
        """Retrieve a particular block of the matrix

            Parameters
            ----------
            block : :class:`tuple`
                tuple (i,j) giving the index of the block

            Returns
            -------
            block
                either an :class:`~numpy:numpy.array`, a :class:`block_sparse` matrix, or a :class:`symmetric_block_sparse` matrix.
        """
        if block in self.submatrix_dict:
            return self.submatrix_dict[block]
        else:
            return 0

    def get_type(self, block):
        """Retrieve the type of a particular block of the matrix

            Parameters
            ----------
            block : :class:`tuple`
                tuple (i,j) giving the index of the block

            Returns
            -------
            block type
                either :class:`~numpy:numpy.array`, :class:`block_sparse`, or :class:`symmetric_block_sparse`.
        """
        if block in self.types:
            return self.types[block]
        else:
            return None

    def transpose(self):
        """Return the transpose of the block-sparse matrix

            Returns
            -------
            :class:`block_sparse`
        """
        if self.submatrices is None or np.sum(self.nonzero) == 0:
            return block_sparse([self.blocks[1], self.blocks[0]],
                                np.zeros((self.n_blocks[1], self.n_blocks[0]), dtype=bool), [])
        else:
            # Copy dictionary
            submatrix_list = list()
            for i in xrange(0, self.n_blocks[1]):
                for j in xrange(0, self.n_blocks[0]):
                    if self.nonzero.T[i, j]:
                        submatrix_list.append(self.submatrix_dict[(j, i)].transpose())
            # Check shapes
            nonzero_count = 0
            for i in xrange(0, self.n_blocks[1]):
                for j in xrange(0, self.n_blocks[0]):
                    if self.nonzero.T[i, j]:
                        ij_shape = (
                        self.blocks[1][i + 1] - self.blocks[1][i], self.blocks[0][j + 1] - self.blocks[0][j])
                        s_shape = submatrix_list[nonzero_count].shape
                        if not s_shape == ij_shape:
                            raise (ValueError(
                                'block ' + str(i) + ',' + str(j) + ' has shape ' + str(s_shape[0]) + ',' + str(
                                    s_shape[1]) + '. Expected ' + str(ij_shape[0]) + ',' + str(ij_shape[1])))
                        nonzero_count += 1
            return block_sparse([self.blocks[1], self.blocks[0]], self.nonzero.T, submatrix_list)

    def frobenius(self, A):
        """Compute the frobenius inner product between the current matrix and matrix A

            Parameters
            ----------
            A : matrix
                matrix A with same dimensions as current matrix. The matrix A can be an :class:`~numpy:numpy.array`,
                :class:`block_sparse` matrix, or :class:`symmetric_block_sparse` matrix. It must have the same block
                structure as the current matrix if the matrix is a :class:`block_sparse` matrix or :class:`symmetric_block_sparse` matrix.

            Returns
            -------
            :class:`float`
                the frobenius inner product between the current matrix and matrix A
        """
        # check blocks match
        if not self.shape == A.shape:
            raise (ValueError('Matrices do not have same shape'))
        if type(A)==np.ndarray:
            A = dense_to_block_sparse(A, self.blocks, False, dtype=A.dtype)
        if not type(A) in [block_sparse, symmetric_block_sparse]:
            raise (ValueError('Other matrix is not block_sparse, symmetric_block_sparse, or numpy array'))
        if not self.n_blocks[0] == A.n_blocks[0]:
            raise (ValueError('Matrices do not have same number of row blocks'))
        if not self.n_blocks[1] == A.n_blocks[1]:
            raise (ValueError('Matrices do not have same number of col blocks'))
        for i in xrange(0, self.n_blocks[0]):
            for j in xrange(0, self.n_blocks[1]):
                if self.blocks[0][i] == A.blocks[0][i] and self.blocks[1][j] == A.blocks[1][j]:
                    pass
                else:
                    raise (ValueError('Blocks do not align'))
        # Compute Frobenius inner product
        frob = 0.0
        if self.submatrices is not None:
            for i in xrange(0, self.n_blocks[0]):
                for j in xrange(0, self.n_blocks[1]):
                    if self.nonzero[i, j] and A.nonzero[i, j]:
                        self_ij = self.get_submatrix((i, j))
                        A_ij = A.get_submatrix((i, j))
                        if type(self_ij) in [block_sparse, symmetric_block_sparse]:
                            if type(A_ij) not in [block_sparse, symmetric_block_sparse]:
                                A_ij = dense_to_block_sparse(A_ij, self_ij.blocks, False, dtype=A_ij.dtype)
                            frob += self_ij.frobenius(A_ij)
                        else:
                            if type(A_ij) in [block_sparse, symmetric_block_sparse]:
                                self_ij = dense_to_block_sparse(self_ij, A_ij.blocks, False, dtype=self_ij.dtype)
                                frob += self_ij.frobenius(A_ij)
                            else:
                                frob += np.sum(self_ij * A_ij)
        return frob

    def norm(self):
        """Compute the frobenius norm of the current matrix

            Returns
            -------
            :class:`float`
                the frobenius norm of the current matrix
        """
        return self.frobenius(self)

    def add(self, A):
        """Matrix addition of a matrix A to current matrix

            Parameters
            ----------
            A : matrix
                matrix A with same dimensions as current matrix. The matrix A can be an :class:`~numpy:numpy.array`,
                :class:`block_sparse` matrix, or :class:`symmetric_block_sparse` matrix. It must have the same block
                structure as the current matrix if the matrix is a :class:`block_sparse` matrix or :class:`symmetric_block_sparse` matrix.

            Returns
            -------
            :class:`block_sparse`
                the block-sparse matrix formed by matrix addition of the current matrix to A
        """
        #
        # matrix multiplication: B=self%*%A
        if not type(A) in [block_sparse, symmetric_block_sparse, np.ndarray]:
            raise (ValueError('Other matrix is not block_sparse or numpy array'))
        if not self.shape == A.shape:
            raise (ValueError('Matrices have shapes ' + str(self.shape[0]) + ',' + str(self.shape[1]) + ' ; ' + str(
                A.shape[0]) + ',' + str(A.shape[1])))
        if type(A) == np.ndarray:
            if np.count_nonzero(A) == 0:
                A_nonzero = np.zeros((self.n_blocks[0], self.n_blocks[1]), dtype=bool)
                A = block_sparse(self.blocks, A_nonzero, [])
            else:
                A = dense_to_block_sparse(A, self.blocks, False, dtype=A.dtype)
        if not self.n_blocks[0] == A.n_blocks[0]:
            raise (ValueError('Matrices do not have same number of row blocks'))
        if not self.n_blocks[1] == A.n_blocks[1]:
            raise (ValueError('Matrices do not have same number of col blocks'))
        for i in xrange(0, self.n_blocks[0]):
            for j in xrange(0, self.n_blocks[1]):
                if self.blocks[0][i] == A.blocks[0][i] and self.blocks[1][j] == A.blocks[1][j]:
                    pass
                else:
                    raise (ValueError('Blocks do not align'))
        if self.submatrices is None:
            return A
        elif A.submatrices is None:
            return self
        else:
            # Determine which blocks in sum will be non-zero
            B_nonzero = np.logical_or(self.nonzero, A.nonzero)
            B_submatrices = list()
            for i in xrange(0, self.n_blocks[0]):
                for j in xrange(0, self.n_blocks[1]):
                    if B_nonzero[i, j]:
                        if np.logical_not(A.nonzero[i, j]):
                            B_submatrices.append(self.get_submatrix((i, j)))
                        elif np.logical_not(self.nonzero[i, j]):
                            B_submatrices.append(A.get_submatrix((i, j)))
                        else:
                            self_ij = self.get_submatrix((i, j))
                            A_ij = A.get_submatrix((i, j))
                            if type(self_ij) in [block_sparse, symmetric_block_sparse]:
                                if type(A_ij) not in [block_sparse, symmetric_block_sparse]:
                                    A_ij = dense_to_block_sparse(A_ij, self_ij.blocks, False, dtype=A_ij.dtype)
                                B_ij = self_ij.add(A_ij)
                            else:
                                if type(A_ij) in [block_sparse, symmetric_block_sparse]:
                                    self_ij = dense_to_block_sparse(self_ij, A_ij.blocks, False, dtype=A_ij.dtype)
                                    B_ij = self_ij.add(A_ij)
                                else:
                                    B_ij = self_ij + A_ij
                            B_submatrices.append(B_ij)

            return block_sparse(self.blocks, B_nonzero, B_submatrices)

    def __add__(self, A):
        """Matrix addition of a matrix A to current matrix

            Parameters
            ----------
            A : matrix
                matrix A with same dimensions as current matrix. The matrix A can be a :class:`~numpy:numpy.array`,
                :class:`block_sparse` matrix, or :class:`symmetric_block_sparse` matrix. It must have the same block
                structure as the current matrix if the matrix is a :class:`block_sparse` matrix or :class:`symmetric_block_sparse` matrix.

            Returns
            -------
            matrix  :class:`block_sparse`
                the block-sparse matrix formed by matrix addition of the current matrix to A
        """
        return self.add(A)

    def dot(self,A):
        """Right multiply the current matrix with another :class:`block_sparse` matrix, :class:`symmetric_block_sparse` matrix, or
        :class:`~numpy:numpy.array`, A.

         Parameters
         ----------
         A : matrix
             matrix A with compatible dimensions and block structure: i.e. the row blocks of A must match the column blocks of the
             current matrix, unless A is an :class:`~numpy:numpy.array`.

         Returns
         -------
         :class:`block_sparse`
             the block-sparse matrix formed by right multiplication of the current matrix by A
        """
        return matmul(self,A)

    def qform(self, y, z=None):
        """Computes quadratic form defined by current matrix and input vectors. Let X be the current :class:`block_sparse` matrix, and y and z column vectors. When it is defined, this computes the quadratic form y'Xz.
           If only y is provided, this computes the quadratic form y'Xy.

            Parameters
            ----------
            y : :class:`~numpy:numpy.array`
                1D numpy array of same length as number of rows of current matrix
            z : :class:`~numpy:numpy.array`
                1D numpy array of same length as number of rows of current matrix. Default :class:`None`.

            Returns
            -------
            :class:`float`
                the value of the quadratic form y'Xz
        """
        #### Compute inner product y'self y
        # check dimensions
        if self.submatrices is None or np.sum(self.nonzero) == 0:
            return 0
        else:
            if z is None:
                z = y
            if y.shape[0] == self.shape[0] and z.shape[0] == self.shape[1]:
                ysy = 0
                for i in xrange(0, self.n_blocks[0]):
                    for j in xrange(0, self.n_blocks[1]):
                        if self.nonzero[i, j]:
                            y_i = y[self.blocks[0][i]:self.blocks[0][i + 1]]
                            z_j = z[self.blocks[1][j]:self.blocks[1][j + 1]]
                            if type(self.get_submatrix((i, j))) in [block_sparse, symmetric_block_sparse]:
                                ysy += self.get_submatrix((i, j)).qform(y_i, z_j)
                            else:
                                ysy += np.dot(y_i.T, self.get_submatrix((i, j)).dot(z_j))

                return ysy
            else:
                raise (ValueError('y shape does not match matrix shape'))

    def to_dense(self):
        """Return the current matrix as a standard (dense) numpy array

            Returns
            -------
            :class:`~numpy:numpy.array`
        """
        out = np.zeros(self.shape, dtype=self.dtype)
        if self.submatrices is None or np.sum(self.nonzero) == 0:
            return out
        else:
            for i in xrange(0, self.n_blocks[0]):
                for j in xrange(0, self.n_blocks[1]):
                    if self.nonzero[i, j]:
                        if type(self.get_submatrix((i, j))) in [block_sparse, symmetric_block_sparse]:
                            out[self.blocks[0][i]:self.blocks[0][i + 1],
                            self.blocks[1][j]:self.blocks[1][j + 1]] = self.get_submatrix((i, j)).to_dense()
                        else:
                            out[self.blocks[0][i]:self.blocks[0][i + 1],
                            self.blocks[1][j]:self.blocks[1][j + 1]] = self.get_submatrix((i, j))
            return out


class symmetric_block_sparse(block_sparse):
    """Define a symmetric block-sparse matrix. Inherits some methods from :class:`block_sparse`.

    Parameters
    ----------
    blocks : :class:`~numpy:numpy.array`
        1D numpy integer array, starting at zero, followed by block boundaries, which are the same for both rows and columns
    nonzero : :class:`~numpy:numpy.array`
        symmetric boolean numpy array with number of rows equal to number of row blocks, which is equal to the number of col blocks.
        If entry [i,j] of nonzero is True, then the corresponding block is non-zero; if it is False, then the corresponding block is zero.
    submatrices : :class:`list`
        list of submatrices for non-zero blocks in row-major order, ignoring lower-triangular blocks; e.g., block (1,1), (1,2), (2,2),...
        Each submatrix can be a :class:`~numpy:numpy.array`, a :class:`block_sparse` matrix, or a :class:`symmetric_block_sparse` matrix.
    dtype : numpy data type object
        Set the default data type for the submatrices. Default :class:`~numpy:numpy.float32`
    row_names : :class:`~numpy:numpy.array`
        numpy array with names of the row-blocks. Default :class:`None`
    col_names : :class:`~numpy:numpy.array`
        numpy array with names of the col-blocks. Default :class:`None`

    Returns
    -------
    :class:`symmetric_block_sparse`
        block-sparse matrix
    """
    def __init__(self, blocks, nonzero, submatrices, dtype=np.float32, row_names=None, col_names=None):
        # Block information
        if blocks.dtype == int and len(blocks.shape) == 1:
            self.n_blocks = list()
            self.blocks = list()
            self.blocks.append(blocks)
            self.blocks.append(blocks)
            self.n_blocks.append(blocks.shape[0] - 1)
            self.n_blocks.append(blocks.shape[0] - 1)
            self.shape = (self.blocks[0][self.n_blocks[0]], self.blocks[1][self.n_blocks[1]])
        else:
            raise (ValueError('block boundaries must a 1D numpy integer array'))
        # non-zero submatrix information
        if row_names is not None:
            if not row_names.shape[0] == self.shape[0]:
                raise (ValueError('Length of row names does not match shape of matrix'))
        if col_names is not None:
            if not col_names.shape[0] == self.shape[1]:
                raise (ValueError('Length of col names does not match shape of matrix'))
        self.row_names = row_names
        self.col_names = col_names
        if nonzero.dtype == bool and len(nonzero.shape) == 2:
            if nonzero.shape[0] == self.n_blocks[0] and nonzero.shape[1] == self.n_blocks[1] and np.allclose(nonzero,
                                                                                                             nonzero.T):
                self.nonzero = nonzero
                self.n_nonzero = np.sum(self.nonzero)
                self.n_nonzero_diag = np.sum(np.diag(self.nonzero))
            else:
                raise (ValueError('Number of blocks does not match non-zero pattern or non-zero matrix not symmetric'))
        else:
            raise (ValueError('non-zero pattern must be boolean array'))
        if len(submatrices) == ((self.n_nonzero - self.n_nonzero_diag) / 2 + self.n_nonzero_diag):
            # Check datatypes
            if len(submatrices) > 0:
                # Dictionary of types of submatrices
                self.types = {}
                # Mapping of nonzero to submatrices
                self.submatrix_dict = {}
                nonzero_count = 0
                for i in xrange(0, self.n_blocks[0]):
                    for j in xrange(i, self.n_blocks[1]):
                        if self.nonzero[i, j]:
                            # Check submatrices of correct dimension
                            ij_shape = (
                            self.blocks[0][i + 1] - self.blocks[0][i], self.blocks[1][j + 1] - self.blocks[1][j])
                            submatrix_type = type(submatrices[nonzero_count])
                            if submatrices[nonzero_count].shape == ij_shape and submatrix_type in [block_sparse,
                                                                                                   symmetric_block_sparse,
                                                                                                   np.ndarray]:
                                self.submatrix_dict[(i, j)] = submatrices[nonzero_count]
                                self.types[(i, j)] = submatrix_type
                                # go to next submatrix
                                nonzero_count += 1
                            else:
                                raise (ValueError(
                                    'block ' + str(i) + ',' + str(j) + ' incorrect dimension or unsupported type'))

                self.submatrices = submatrices
                self.dtype = dtype
            else:
                self.types = None
                self.submatrices = None
                self.submatrix_dict = None
                self.dtype = dtype
        else:
            raise (ValueError('Number of submatrices does not match number of non-zero blocks'))

    def get_submatrix(self, block):
        """Retrieve a particular block of the matrix

            Parameters
            ----------
            block : :class:`tuple`
                tuple (i,j) giving the index of the block

            Returns
            -------
            block
                either a :class:`~numpy:numpy.array`, a :class:`block_sparse` matrix, or a :class:`symmetric_block_sparse` matrix.
        """
        if block in self.submatrix_dict:
            return self.submatrix_dict[block]
        elif (block[1], block[0]) in self.submatrix_dict:
            return self.submatrix_dict[(block[1], block[0])].transpose()
        else:
            return 0

    def get_type(self, block):
        """Retrieve the type of a particular block of the matrix

            Parameters
            ----------
            block : :class:`tuple`
                tuple (i,j) giving the index of the block

            Returns
            -------
            block type
                either :class:`~numpy:numpy.array`, :class:`block_sparse` matrix, or :class:`symmetric_block_sparse` matrix.
        """
        if block in self.types:
            return self.types[block]
        elif (block[1], block[0]) in self.types:
            return self.types[(block[1], block[0])]
        else:
            return None

    def add(self, A):
        """Matrix addition of a matrix A to current matrix.

            Parameters
            ----------
            A : matrix
                matrix A with same dimensions as current matrix. The matrix A can be a :class:`~numpy:numpy.array`,
                :class:`block_sparse` matrix, or :class:`symmetric_block_sparse` matrix. It must have the same block
                structure as the current matrix if the matrix is a :class:`block_sparse` matrix or :class:`symmetric_block_sparse` matrix.

            Returns
            -------
            matrix
                If A is :class:`symmetric_block_sparse`, returns a :class:`symmetric_block_sparse` matrix. Otherwise, returns
                a :class:`block_sparse` matrix.
        """
        if not type(A) in [block_sparse, symmetric_block_sparse, np.ndarray]:
            raise (ValueError('Other matrix is not block_sparse or numpy array'))
        if not self.shape == A.shape:
            raise (ValueError('Matrices have shapes ' + str(self.shape[0]) + ',' + str(self.shape[1]) + ' ; ' + str(
                A.shape[0]) + ',' + str(A.shape[1])))
        if type(A) == block_sparse:
            return A.add(self)
        if type(A) == np.ndarray:
            if np.count_nonzero(A) == 0:
                return self
            elif np.allclose(A, A.T):
                A = dense_to_block_sparse(A, self.block, True, dtype=A.dtype)
            else:
                A = dense_to_block_sparse(A, self.blocks, False, dtype=A.dtype)
                return A.add(self)
        if not self.n_blocks[0] == A.n_blocks[0]:
            raise (ValueError('Matrices do not have same number of row blocks'))
        if not self.n_blocks[1] == A.n_blocks[1]:
            raise (ValueError('Matrices do not have same number of col blocks'))
        for i in xrange(0, self.n_blocks[0]):
            for j in xrange(0, self.n_blocks[1]):
                if self.blocks[0][i] == A.blocks[0][i] and self.blocks[1][j] == A.blocks[1][j]:
                    pass
                else:
                    raise (ValueError('Blocks do not align'))
        if self.submatrices is None or np.sum(self.nonzero) == 0:
            return A
        elif A.submatrices is None or np.sum(A.nonzero) == 0:
            return self
        else:
            # Determine which blocks in sum will be non-zero
            B_nonzero = np.logical_or(self.nonzero, A.nonzero)
            B_submatrices = list()
            for i in xrange(0, self.n_blocks[0]):
                for j in xrange(i, self.n_blocks[1]):
                    if B_nonzero[i, j]:
                        if np.logical_not(A.nonzero[i, j]):
                            B_submatrices.append(self.get_submatrix((i, j)))
                        elif np.logical_not(self.nonzero[i, j]):
                            B_submatrices.append(A.get_submatrix((i, j)))
                        else:
                            self_ij = self.get_submatrix((i, j))
                            A_ij = A.get_submatrix((i, j))
                            if type(self_ij) in [block_sparse, symmetric_block_sparse]:
                                if type(A_ij) not in [block_sparse, symmetric_block_sparse]:
                                    A_ij = dense_to_block_sparse(A_ij, self_ij.blocks, False, dtype=A_ij.dtype)
                                B_ij = self_ij.add(A_ij)
                            else:
                                if type(A_ij) in [block_sparse, symmetric_block_sparse]:
                                    self_ij = dense_to_block_sparse(self_ij, A_ij.blocks, False, dtype=self_ij.dtype)
                                    B_ij = self_ij.add(A_ij)
                                else:
                                    B_ij = self_ij + A_ij
                            B_submatrices.append(B_ij)

            return symmetric_block_sparse(self.blocks[0], B_nonzero, B_submatrices)

    def __add__(self, A):
        return self.add(A)

    def qform(self, y, z=None):
        """Let X be the current :class:`symmetric_block_sparse` matrix, and y and z column vectors. When it is defined, this computes the quadratic form y'Xz.
           If only y is provided, this computes the quadratic form y'Xy.

            Parameters
            ----------
            y : :class:`~numpy:numpy.array`
                1D numpy array of same length as number of rows of current matrix
            z : :class:`~numpy:numpy.array`
                1D numpy array of same length as number of rows of current matrix. Default :class:`None`.

            Returns
            -------
            :class:`float`
                the value of the quadratic form y'Xz
        """
        if self.submatrices is None or np.sum(self.nonzero) == 0:
            return 0
        else:
            if z is None:
                z = y
            if y.shape[0] == self.shape[0] and z.shape[0] == self.shape[1]:
                ysy = 0
                for i in xrange(0, self.n_blocks[0]):
                    for j in xrange(i, self.n_blocks[1]):
                        if self.nonzero[i, j]:
                            if i == j:
                                scale = 1
                            else:
                                scale = 2
                            y_i = y[self.blocks[0][i]:self.blocks[0][i + 1]]
                            z_j = z[self.blocks[1][j]:self.blocks[1][j + 1]]
                            if type(self.get_submatrix((i, j))) in [block_sparse, symmetric_block_sparse]:
                                ysy += scale * self.get_submatrix((i, j)).qform(y_i, z_j)
                            else:
                                ysy += scale * np.dot(y_i.T, self.get_submatrix((i, j)).dot(z_j))

                return ysy
            else:
                raise (ValueError('y shape does not match matrix shape'))

    def transpose(self):
        """Return the transpose of the symmetric block-sparse matrix

            Returns
            -------
            :class:`symmetric_block_sparse`
                the current matrix, as it is symmetric
        """
        return self

    def to_dense(self):
        """Return the current matrix as a standard (dense) numpy array

            Returns
            -------
            :class:`~numpy:numpy.array`
        """
        out = np.zeros(self.shape, dtype=self.dtype)
        if self.submatrices is None or np.sum(self.nonzero) == 0:
            return out
        else:
            for i in xrange(0, self.n_blocks[0]):
                for j in xrange(i, self.n_blocks[1]):
                    if self.nonzero[i, j]:
                        if type(self.get_submatrix((i, j))) in [block_sparse, symmetric_block_sparse]:
                            out[self.blocks[0][i]:self.blocks[0][i + 1],
                            self.blocks[1][j]:self.blocks[1][j + 1]] = self.get_submatrix((i, j)).to_dense()
                        else:
                            out[self.blocks[0][i]:self.blocks[0][i + 1],
                            self.blocks[1][j]:self.blocks[1][j + 1]] = self.get_submatrix((i, j))
            out = out + out.T
            for i in xrange(0, self.n_blocks[0]):
                out[self.blocks[0][i]:self.blocks[0][i + 1], self.blocks[1][i]:self.blocks[1][i + 1]] = out[
                                                                                                        self.blocks[0][
                                                                                                            i]:
                                                                                                        self.blocks[0][
                                                                                                            i + 1],
                                                                                                        self.blocks[1][
                                                                                                            i]:
                                                                                                        self.blocks[1][
                                                                                                            i + 1]] / 2.0
            return out


def dense_to_block_sparse(dense, blocks, symmetric, dtype=np.float64):
    """Convert a standard (dense) numpy array into a :class:`block_sparse` or
       a :class:`symmetric_block_sparse` matrix. Note this simply imposes a block structure onto
       the matrix so that it can interact with other block matrices. It does not take advantage
       of any sparsity in the input matrix.

        Parameters
        ----------
        dense : :class:`~numpy:numpy.array`
            input matrix
        blocks : :class:`list`
            list of [row_block_boundaries,col_block_boundaries], where each of row_block_boundaries and col_block_boundaries is
            a 1D :class:`~numpy:numpy.array` of integers, beginning with 0, followed by the end boundaries of each block in increasing order
        symmetric : :class:`bool`
            if True, returns a :class:`symmetric_block_sparse` matrix; if False, returns a :class:`block_sparse` matrix
        dtype : numpy data type
            the default data type of the returned matrix

        Returns
        -------
        matrix
            the current matrix as a :class:`block_sparse` or a :class:`symmetric_block_sparse` matrix
    """
    n_row_blocks = blocks[0].shape[0] - 1
    n_col_blocks = blocks[0].shape[0] - 1
    if not blocks[0][n_row_blocks] == dense.shape[0]:
        raise (ValueError('blocks do not match matrix dimensions'))
    if not blocks[1][n_col_blocks] == dense.shape[1]:
        raise (ValueError('blocks do not match matrix dimensions'))
    if symmetric and not np.allclose(blocks[0], blocks[1]):
        raise (ValueError('Asymmetric block structure not allowed for symmetric block-sparse matrix'))
    nonzero = np.ones((n_row_blocks, n_col_blocks), dtype=bool)
    submatrices = list()
    for i in xrange(0, n_row_blocks):
        if symmetric:
            for j in xrange(i, n_col_blocks):
                submatrices.append(dense[blocks[0][i]:blocks[0][i + 1], blocks[1][j]:blocks[1][j + 1]])
        else:
            for j in xrange(0, n_col_blocks):
                submatrices.append(dense[blocks[0][i]:blocks[0][i + 1], blocks[1][j]:blocks[1][j + 1]])
    if symmetric:
        return symmetric_block_sparse(blocks[0], nonzero, submatrices, dtype=dtype)
    else:
        return block_sparse(blocks, nonzero, submatrices, dtype=dtype)

def matmul(X, A):
    """Matrix multiplication between :class:`block_sparse` and :class:`symmetric_block_sparse` matrices, as well
        as matrix multiplication between a :class:`block_sparse` or :class:`symmetric_block_sparse` matrix and an :class:`~numpy:numpy.array`.

        Parameters
        ----------
        X : matrix
            The matrix X can be a :class:`block_sparse` matrix, a :class:`symmetric_block_sparse` matrix, or a :class:`~numpy:numpy.array`.

        A : matrix
            The matrix A can be a :class:`block_sparse` matrix, a :class:`symmetric_block_sparse` matrix, or a :class:`~numpy:numpy.array`.
            Note that the number of rows of A must match the number of columns of X. Furthermore, if X and A are both
            :class:`block_sparse` or :class:`symmetric_block_sparse`, then the column blocks of X must match the row blocks of A.

        Returns
        -------
        :class:`block_sparse`
            the block-sparse matrix formed by matrix multiplication XA
    """
    if not X.shape[1]==A.shape[0]:
        raise(ValueError('Matrices have incompatible dimensions'))
    if type(A)==np.ndarray and type(X)==np.ndarray:
        return np.dot(X,A)
    elif type(A)==np.ndarray and type(X) in [block_sparse, symmetric_block_sparse]:
        A_blocks = [X.blocks[1],np.array(0,A.shape[1],dtype=int)]
        A = dense_to_block_sparse(A, A_blocks, False, dtype = A.dtype)
    elif type(X)==np.ndarray and type(A) in [block_sparse, symmetric_block_sparse]:
        X_blocks = [np.array(0,X.shape[0],dtype=int),A.blocks[0]]
        X = dense_to_block_sparse(X, X_blocks, False, dtype = X.dtype)
    elif type(X) in [block_sparse, symmetric_block_sparse] and type(A) in [block_sparse, symmetric_block_sparse]:
        pass
    else:
        raise(ValueError('Usupported matrix types'))
    # Check block structure matches
    for i in xrange(0, X.n_blocks[1]):
        if not X.blocks[1][i] == A.blocks[0][i]:
            raise (ValueError(
                'X col block boundary at ' + str(X.blocks[1][i]) + ' and A row block boundary at ' + str(
                    A.blocks[0][i])))
    if X.submatrices is None or np.sum(X.nonzero) == 0 or A.submatrices is None or np.sum(A.nonzero) == 0:
        return block_sparse([X.blocks[0], A.blocks[1]], np.zeros((X.n_blocks[0], A.n_blocks[1]), dtype=bool),
                            [])
    # Compute dot product
    if not X.dtype == A.dtype:
        raise (Warning('Data types do not match: ' + str(X.dtype)))
    else:
        # Determine which blocks in product will be non-zero
        B_nonzero = X.nonzero.dot(A.nonzero)
        B_submatrices = list()
        B_blocks = [X.blocks[0], A.blocks[1]]
        for i in xrange(0, X.n_blocks[0]):
            for j in xrange(0, A.n_blocks[1]):
                if B_nonzero[i, j]:
                    # Identify if resulting i,j submatrix should be block sparse
                    B_ij = np.zeros(
                        (X.blocks[0][i + 1] - X.blocks[0][i], A.blocks[1][j + 1] - A.blocks[1][j]),
                        dtype=X.dtype)
                    for k in xrange(0, X.n_blocks[1]):
                        if X.nonzero[i, k] and A.nonzero[k, j]:
                            X_ik = X.get_submatrix((i, k))
                            A_kj = A.get_submatrix((k, j))
                            if type(X_ik) in [block_sparse, symmetric_block_sparse]:
                                if type(A_kj) not in [block_sparse, symmetric_block_sparse]:
                                    A_kj = dense_to_block_sparse(A_kj, [X_ik.blocks[1], X_ik.blocks[0]],
                                                                 False, dtype=A_kj.dtype)
                                    B_ijk = X_ik.dot(A_kj)
                                    B_ij = B_ijk.add(B_ij)
                                B_ijk = X_ik.dot(A_kj)
                                if not B_ijk.shape == B_ij.shape:
                                    raise (ValueError(
                                        'Block ' + str(i) + ',' + str(j) + ' should have shape ' + str(
                                            X.blocks[0][i + 1] - X.blocks[0][i]) + ',' + str(
                                            A.blocks[1][j + 1] - A.blocks[1][j]) + '.  B_ijk has shape ' + str(
                                            B_ijk.shape[0]) + ',' + str(B_ijk.shape[1]) + ' k=' + str(
                                            k) + ' i=' + str(i) + ' j=' + str(j) + ' X has shape ' + str(
                                            X_ik.shape[0]) + ',' + str(X_ik.shape[1]) + ' A has shape ' + str(
                                            A_kj.shape[0]) + ',' + str(A_kj.shape[1])))
                                B_ij = B_ijk.add(B_ij)
                            else:
                                if type(A_kj) in [block_sparse, symmetric_block_sparse]:
                                    X_ik = dense_to_block_sparse(X_ik, A_kj.blocks, False, dtype=A_kj.dtype)
                                    B_ij = X_ik.dot(A_kj).add(B_ij)
                                else:
                                    if type(B_ij) == np.ndarray:
                                        B_ij += np.dot(X_ik, A_kj)
                                    else:
                                        B_ij = B_ij.add(np.dot(X_ik, A_kj))
                    B_submatrices.append(B_ij)

    return block_sparse(B_blocks, B_nonzero, B_submatrices)

