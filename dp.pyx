import numpy as np
cimport numpy as np
cimport cython

NP_FLOAT = np.float64
NP_INT = np.int32

ctypedef np.float64_t NP_FLOAT_t
ctypedef np.int32_t NP_INT_t

cdef int get_step(int k):
	return 0 if k%2==0 else (k+1)/2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dp(float[:,:] Y, float[:,:] C, int exactly_one=True, bg_cost=0):
	cdef int T = Y.shape[0]
	cdef int K = Y.shape[1]
	cdef int K_ext = 2*K+1

	cdef NP_FLOAT_t[:,:] L = -np.ones([T+1,K_ext], dtype=NP_FLOAT)
	cdef NP_INT_t[:,:] P = -np.ones([T+1,K_ext], dtype=NP_INT)
	L[0,0] = 0
	P[0,0] = 0

	cdef int opt_label
	cdef double opt_value
	cdef int j,t,s
	cdef NP_FLOAT_t[:] Lt
	cdef NP_INT_t[:] Pt
	for t in range(1,T+1):
		Lt = L[t-1,:]
		Pt = P[t-1,:]
		for k in range(K_ext):
			s = get_step(k)

			opt_label = -1

			j = k
			if (opt_label==-1 or opt_value>Lt[j]) and Pt[j]!=-1 and (s==0 or not exactly_one):
				opt_label = j
				opt_value = Lt[j]

			j = k-1
			if j>=0 and (opt_label==-1 or opt_value>Lt[j]) and Pt[j]!=-1:
				opt_label = j
				opt_value = L[t-1][j]

			if s!=0:
				j = k-2
				if j>=0 and (opt_label==-1 or opt_value>Lt[j]) and Pt[j]!=-1:
					opt_label = j
					opt_value = Lt[j]

			if s!=0:
				L[t,k] = opt_value + C[t-1][s-1]
			else:
				L[t,k] = opt_value + bg_cost
			P[t,k] = opt_label

	for t in range(T):
		for k in range(K):
			Y[t,k] = 0
	if (L[T,K_ext-1] < L[T,K_ext-2] or (P[T,K_ext-2]==-1)):
		k = K_ext-1
	else:
		k = K_ext-2
	for t in range(T,0,-1):
		s = get_step(k)
		if s > 0:
			Y[t-1,s-1] = 1
		k = P[t,k]