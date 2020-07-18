import numpy as np

def triu_idx(N):
    rows, columns = np.triu_indices(N)
    return rows * N + columns

def vech(mat):
    return mat.T.take(triu_idx(len(mat)))

def unvech(mat):
    rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(mat)))
    rows = int(np.round(rows))
    out = np.zeros((rows, rows))
    out[np.triu_indices(rows)] = mat
    out = out.T + out
    out[np.diag_indices(rows)] /= 2
    return out

def pos_def(mat):
    try:
        np.linalg.cholesky(mat)
        return 1 
    except np.linalg.linalg.LinAlgError:
        return -1

def num_derivs(f, x, h=1e-06):
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