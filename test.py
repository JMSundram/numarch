import numarch as narch
import numpy as np
import time

# Parameter matrices
Omega = 0.8
A     = 0.5

# Generate data
X       = []
OMEGA_T = [Omega]
T = 2000
for t in range(T):
    X.append(np.random.normal(0, np.sqrt(OMEGA_T[t])))
    OMEGA_T.append(Omega + A * X[t] * X[t] * A)
X = np.array(X).reshape(T, 1)

# Fit BEKK-ARCH model
mod = narch.arch('bekk-arch')
t0 = time.time()
for i in range(1): mod.fit(X, se=True, std=False)
t1 = time.time()
print(t1-t0)

mod = narch.arch('bekk-arch')
t0 = time.time()
for i in range(1): mod.fit(X, se=True)
t1 = time.time()
print(t1-t0)

import pandas as pd
df = pd.DataFrame(X)
df.to_excel('test.xlsx')