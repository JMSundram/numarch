# numarch

## About
A python program for numerical maximum likelihood estimation of multivariate autoregressive conditional heteroskedasticity (ARCH) models. Implements the (multivariate) BEKK-ARCH and BEKK-GARCH models, which have the univariate ARCH(1) and GARCH(1,1) models as special cases.

## Minimal working example with generated data
```python
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
    X.append(np.random.multivariate_normal([0, 0], OMEGA_T[t]).reshape(2, 1))
    OMEGA_T.append(Omega + A @ X[t] @ X[t].T @ A.T)
X = np.array(X).reshape(T, 2)

# Fit BEKK-ARCH model
mod = narch.arch('bekk-arch')
mod.fit(X)
```
