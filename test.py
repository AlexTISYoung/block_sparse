import block_sparse, unittest
import numpy as np

precision = 5

n_per_test = 10**4

from numpy import testing
def dense_and_block(row_blocks,col_blocks,symmetrical=False,nonzeros=None):
	        n_row_blocks=row_blocks.shape[0]-1	
	        n_col_blocks=col_blocks.shape[0]-1	
		row_size=row_blocks[n_row_blocks]
		col_size=col_blocks[n_col_blocks]
		X=np.random.randn(row_size*col_size).reshape((row_size,col_size))
		if symmetrical:
			if not np.array_equal(row_blocks,col_blocks):
				raise(ValueError('Row blocks must be the same as column blocks for symmetrical block-sparse matrix'))
			X=X+X.T
		# Randomly set some to be zero
		if nonzeros is None:
			nonzeros=np.array(np.random.randint(0,2,n_row_blocks*n_col_blocks).reshape(n_row_blocks,n_col_blocks),dtype=bool)
			if symmetrical:
				nonzeros=nonzeros+nonzeros.T
		for i in xrange(0,n_row_blocks):
			for j in xrange(0,n_col_blocks):
				if not nonzeros[i,j]:
					X[row_blocks[i]:row_blocks[i+1],col_blocks[j]:col_blocks[j+1]]=0
		submatrices=list()
		for i in xrange(0,n_row_blocks):
			if symmetrical:
				jstart=i
			else:
				jstart=0
			for j in xrange(jstart,n_col_blocks):
				if nonzeros[i,j]:
					X_ij=X[row_blocks[i]:row_blocks[i+1],col_blocks[j]:col_blocks[j+1]]
					submatrices.append(X_ij)
		if symmetrical:
			Z=block_sparse.symmetric_block_sparse(row_blocks,nonzeros,submatrices)
		else:
			Z=block_sparse.block_sparse([row_blocks,col_blocks],nonzeros,submatrices)
		return [X,Z]

def dense_and_block_higher(higher_blocks,blocks,symmetrical=False,lower_nonzeros=None):
	n_blocks=higher_blocks.shape[0]-1	
	row_size=higher_blocks[n_blocks]
	nonzeros=np.array(np.random.randint(0,2,n_blocks**2).reshape(n_blocks,n_blocks),dtype=bool)
	if symmetrical:
		nonzeros=nonzeros+nonzeros.T
	#nonzeros=np.array([True,True,False,False]).reshape((2,2))
	#nonzeros=np.zeros((n_blocks,n_blocks),dtype=bool)
	submatrices=list()
	X=np.zeros((row_size,row_size))
	for i in xrange(0,n_blocks):
		if symmetrical:
			jstart=i
		else:
			jstart=0
		for j in xrange(jstart,n_blocks):
			if nonzeros[i,j]:
				if not i==j:
					symmetrical_ij=False
				else:
					symmetrical_ij=symmetrical
				X_ij,Z_ij=dense_and_block(blocks[i],blocks[j],symmetrical_ij,lower_nonzeros)
				X[higher_blocks[i]:higher_blocks[i+1],higher_blocks[j]:higher_blocks[j+1]]=X_ij
				submatrices.append(Z_ij)
	if symmetrical:
		X=X+X.T
		for i in xrange(0,n_blocks):
			X[higher_blocks[i]:higher_blocks[i+1],higher_blocks[i]:higher_blocks[i+1]]=X[higher_blocks[i]:higher_blocks[i+1],higher_blocks[i]:higher_blocks[i+1]]/2.0
		Z=block_sparse.symmetric_block_sparse(higher_blocks,nonzeros,submatrices)
	else:	
		Z=block_sparse.block_sparse([higher_blocks,higher_blocks],nonzeros,submatrices)
	return [X,Z]
		
class test_block_sparse_methods(unittest.TestCase):

	def test_init_to_dense(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			Z=Z.to_dense()
			testing.assert_array_almost_equal(X,Z,decimal = precision)

	def test_frobenius(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			A,B=dense_and_block_higher(higher_blocks,lower_blocks)
			XA_F=np.sum(X*A)
			ZB_F=Z.frobenius(B)
			testing.assert_almost_equal(XA_F,ZB_F,decimal = precision)
	
	def test_transpose(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			Z=Z.transpose()
			testing.assert_almost_equal(X.transpose(),Z.to_dense(),decimal = precision)
	
	def test_add(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
                        X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			A,B=dense_and_block_higher(higher_blocks,lower_blocks)
			XA=X+A
			ZB=Z.add(B)
			ZB=ZB.to_dense()
                        testing.assert_almost_equal(XA,ZB,decimal = precision)
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
                        X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			zero=np.zeros(X.shape)
			Z=Z.add(zero)
			testing.assert_almost_equal(X,Z.to_dense(),decimal = precision)
		
	def test_dot(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,15,30],dtype=int)]
			#nonzeros=np.array(np.random.randint(0,2,4).reshape(2,2),dtype=bool)
			#nonzeros=np.array([False,False,False,False]).reshape((2,2))
			#np.save('nonzeros.npy',nonzeros)
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			A,B=dense_and_block_higher(higher_blocks,lower_blocks)
			XA=X.dot(A)
			ZB=Z.dot(B)
			ZB=ZB.to_dense()
			#np.save('XA.npy',XA)
			#np.save('ZB.npy',ZB)
			testing.assert_array_almost_equal(XA,ZB,decimal = precision)
		 
	def test_qform(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			#nonzeros=np.array([False,False,False,False]).reshape((2,2))
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			y=np.random.randn(X.shape[0])
			yXy=np.dot(y.T,X.dot(y))
			yZy=Z.qform(y)
			testing.assert_almost_equal(yXy,yZy,decimal = precision)
	
	def test_init_to_dense_s(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks,True)
			Z=Z.to_dense()
			testing.assert_array_almost_equal(X,Z,decimal = precision)

	def test_frobenius_s(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks,True)
			A,B=dense_and_block_higher(higher_blocks,lower_blocks,True)
			XA_F=np.sum(X*A)
			ZB_F=Z.frobenius(B)
			testing.assert_almost_equal(XA_F,ZB_F,decimal = precision)
	
	def test_transpose_s(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks,True)
			Z=Z.transpose()
			testing.assert_almost_equal(X.transpose(),Z.to_dense(),decimal = precision)
	
	def test_add_s(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
                        X,Z=dense_and_block_higher(higher_blocks,lower_blocks,True)
			A,B=dense_and_block_higher(higher_blocks,lower_blocks,True)
			XA=X+A
			ZB=Z.add(B)
			ZB=ZB.to_dense()
                        testing.assert_almost_equal(XA,ZB,decimal = precision)
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
                        X,Z=dense_and_block_higher(higher_blocks,lower_blocks)
			zero=np.zeros(X.shape)
			Z=Z.add(zero)
			testing.assert_almost_equal(X,Z.to_dense(),decimal = precision)
		
	def test_dot_s(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,15,30],dtype=int)]
			#nonzeros=np.array(np.random.randint(0,2,4).reshape(2,2),dtype=bool)
			#nonzeros=np.array([False,False,False,False]).reshape((2,2))
			#np.save('nonzeros.npy',nonzeros)
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks,True)
			A,B=dense_and_block_higher(higher_blocks,lower_blocks,True)
			XA=X.dot(A)
			ZB=Z.dot(B)
			ZB=ZB.to_dense()
			#np.save('XA.npy',XA)
			#np.save('ZB.npy',ZB)
			testing.assert_array_almost_equal(XA,ZB,decimal = precision)
		 
	def test_qform_s(self):
		for i in xrange(0,n_per_test):
			higher_blocks=np.array([0,20,50,100],dtype=int)
			lower_blocks=[np.array([0,5,20],dtype=int),np.array([0,10,30],dtype=int),np.array([0,1,30,50],dtype=int)]
			#nonzeros=np.array([False,False,False,False]).reshape((2,2))
			X,Z=dense_and_block_higher(higher_blocks,lower_blocks,True)
			y=np.random.randn(X.shape[0])
			yXy=np.dot(y.T,X.dot(y))
			yZy=Z.qform(y)
			testing.assert_almost_equal(yXy,yZy,decimal = precision)

if  __name__=='__main__':
	unittest.main()
