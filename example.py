import numarch as narch
import numpy as np

# Parameter matrices
Omega = np.array([[0.8, 0.5],
                  [0.5, 0.8]])
A     = np.array([[0.5, 0.1],
                  [0.3, 0.1]])

# Generate data
X       = []
OMEGA_T = [Omega]
T = 2000
for t in range(T):
    X.append(np.random.multivariate_normal([0, 0], OMEGA_T[t], 1))
    OMEGA_T.append(Omega + A @ X[t].reshape(2,1) @ X[t].reshape(1,2) @ A.T)
X = np.array(X).reshape(T, 2)

# Fit BEKK-ARCH model
mod = narch.arch('bekk-arch')
mod.fit(X)