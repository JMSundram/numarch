import numpy as np

def triu_idx(N):
    '''Return indices of upper triangular elements of N-dimensional matrix'''
    rows, columns = np.triu_indices(N)
    return rows * N + columns

def vech(mat):
    '''Vec-half operator which returns above-diagonal matrix elements'''
    return mat.T.take(triu_idx(len(mat)))

def unvech(mat):
    '''Inverse of vec-half operator'''
    rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(mat)))
    rows = int(np.round(rows))
    out = np.zeros((rows, rows))
    out[np.triu_indices(rows)] = mat
    out = out.T + out
    out[np.diag_indices(rows)] /= 2
    return out

def pos_def(mat):
    '''Check if matrix is positive definite'''  
    try:
        np.linalg.cholesky(mat)
        return 1 
    except np.linalg.linalg.LinAlgError:
        return -1

def num_derivs(f, x, h=1e-08):
    '''Estimate derivatives of vector or scalar function
    
    Keyword arguments:
    f -- scalar or vector function
    x -- point to calculate derivative in
    h -- step size
    '''
    if type(f(x)) == list:
        scalar_f = False
    else:
        scalar_f = True
    derivs = []
    for i in range(len(x)):
        xm = x.copy()
        xp = x.copy()
        xm[i] = xm[i] - h
        xp[i] = xp[i] + h
        fxp = f(xp)
        fxm = f(xm)
        if scalar_f:
            derivs.append((fxp - fxm)/(2*h))
        else:
            derivs.append([(fxp[j] - fxm[j])/(2*h) for j in range(len(fxp))])
    return np.array(derivs).T