import numpy as np

class block_sparse(object):
    """
    A square matrix with block structure
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
                self.dtype = dtype
            else:
                self.types = None
                self.submatrices = None
                self.submatrix_dict = None
                self.dtype = dtype
        else:
            raise (ValueError('Number of submatrices does not match number of non-zero blocks'))

    def get_submatrix(self, block):
        if block in self.submatrix_dict:
            return self.submatrix_dict[block]
        else:
            return 0

    def get_type(self, block):
        if block in self.types:
            return self.types[block]
        else:
            return None

    def transpose(self):
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
        # Compute the frobenius inner product between the current matrix and matrix A
        # check blocks match
        if not type(A) in [block_sparse, symmetric_block_sparse]:
            raise (ValueError('Other matrix is not block_sparse'))
        if not self.shape == A.shape:
            raise (ValueError('Matrices do not have same shape'))
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
        frob = 0
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

    def add(self, A):
        # Add a block-sparse matrix A to self
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

    def __add__(self, other):
        return self.add(other)

    def dot(self, A):
        # matrix multiplication: B=self%*%A
        if not type(A) in [block_sparse, symmetric_block_sparse]:
            raise (ValueError('Other matrix is not block_sparse'))
        if not self.n_blocks[1] == A.n_blocks[0]:
            raise (ValueError('Number of col blocks of first matrix not equal to number of row blocks of second'))
        for i in xrange(0, self.n_blocks[1]):
            if not self.blocks[1][i] == A.blocks[0][i]:
                raise (ValueError(
                    'self col block boundary at ' + str(self.blocks[1][i]) + ' and A row block boundary at ' + str(
                        A.blocks[0][i])))
        if self.submatrices is None or np.sum(self.nonzero) == 0 or A.submatrices is None or np.sum(A.nonzero) == 0:
            return block_sparse([self.blocks[0], A.blocks[1]], np.zeros((self.n_blocks[0], A.n_blocks[1]), dtype=bool),
                                [])
        if not self.dtype == A.dtype:
            raise (Warning('Data types do not match: ' + str(self.dtype)))
        else:
            # Determine which blocks in product will be non-zero
            B_nonzero = self.nonzero.dot(A.nonzero)
            B_submatrices = list()
            B_blocks = [self.blocks[0], A.blocks[1]]
            for i in xrange(0, self.n_blocks[0]):
                for j in xrange(0, A.n_blocks[1]):
                    if B_nonzero[i, j]:
                        # Identify if resulting i,j submatrix should be block sparse
                        B_ij = np.zeros(
                            (self.blocks[0][i + 1] - self.blocks[0][i], A.blocks[1][j + 1] - A.blocks[1][j]),
                            dtype=self.dtype)
                        for k in xrange(0, self.n_blocks[1]):
                            if self.nonzero[i, k] and A.nonzero[k, j]:
                                self_ik = self.get_submatrix((i, k))
                                A_kj = A.get_submatrix((k, j))
                                if type(self_ik) in [block_sparse, symmetric_block_sparse]:
                                    if type(A_kj) not in [block_sparse, symmetric_block_sparse]:
                                        A_kj = dense_to_block_sparse(A_kj, [self_ik.blocks[1], self_ik.blocks[0]],
                                                                     False, dtype=A_kj.dtype)
                                        B_ijk = self_ik.dot(A_kj)
                                        B_ij = B_ijk.add(B_ij)
                                    B_ijk = self_ik.dot(A_kj)
                                    if not B_ijk.shape == B_ij.shape:
                                        raise (ValueError(
                                            'Block ' + str(i) + ',' + str(j) + ' should have shape ' + str(
                                                self.blocks[0][i + 1] - self.blocks[0][i]) + ',' + str(
                                                A.blocks[1][j + 1] - A.blocks[1][j]) + '.  B_ijk has shape ' + str(
                                                B_ijk.shape[0]) + ',' + str(B_ijk.shape[1]) + ' k=' + str(
                                                k) + ' i=' + str(i) + ' j=' + str(j) + ' self has shape ' + str(
                                                self_ik.shape[0]) + ',' + str(self_ik.shape[1]) + ' A has shape ' + str(
                                                A_kj.shape[0]) + ',' + str(A_kj.shape[1])))
                                    B_ij = B_ijk.add(B_ij)
                                else:
                                    if type(A_kj) in [block_sparse, symmetric_block_sparse]:
                                        self_ik = dense_to_block_sparse(self_ik, A_kj.blocks, False, dtype=A_kj.dtype)
                                        B_ij = self_ik.dot(A_kj).add(B_ij)
                                    else:
                                        if type(B_ij) == np.ndarray:
                                            B_ij += np.dot(self_ik, A_kj)
                                        else:
                                            B_ij = B_ij.add(np.dot(self_ik, A_kj))
                        B_submatrices.append(B_ij)

            return block_sparse(B_blocks, B_nonzero, B_submatrices)

    def qform(self, y, z=None):
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
            raise (ValueError('block boundaries must be given as a list of 1D numpy integer arrays'))
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
        if block in self.submatrix_dict:
            return self.submatrix_dict[block]
        elif (block[1], block[0]) in self.submatrix_dict:
            return self.submatrix_dict[(block[1], block[0])].transpose()
        else:
            return 0

    def get_type(self, block):
        if block in self.types:
            return self.types[block]
        elif (block[1], block[0]) in self.types:
            return self.types[(block[1], block[0])]
        else:
            return None

    def add(self, A):
        # Add a block-sparse matrix A to self
        # matrix multiplication: B=self%*%A
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

    def __radd__(self, A):
        return self.add(A)

    def qform(self, y, z=None):
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
        return self

    def to_dense(self):
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