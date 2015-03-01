import numpy as np
import matplotlib.pyplot as plt

N = 1000
ts = np.linspace(0, 5, N)

N1 = 200
N2 = 200
N3 = 200
N4 = 200
N5 = 200
xs1 = np.sin(100 * np.pi * ts[:N1])
xs2 = np.sin(20 * np.pi * ts[N1:N2+N1])
xs3 = np.cos(100 * np.pi * ts[N2+N1:N3+N2+N1])
xs4 = np.sin(20 * np.pi * ts[N3+N2+N1:N4+N3+N2+N1])
xs5 = np.sin(100 * np.pi * ts[N4+N3+N2+N1:N])
xs = np.hstack([xs1, xs2, xs3, xs4, xs5])

rs = np.random.random(N)
xs = xs + rs * 0.05


# Change-Point Detection of Climate Time Series by Nonparametric Method
# http://www.iaeng.org/publication/WCECS2010/WCECS2010_pp445-448.pdf
# n = w, gamma = n/2
w = 40
n = w
gamma = int(n/2)

def create_S(x, t, w):
    return x[t-w:t]

def create_H(x, t, w, n):
    H = np.zeros((w, n), dtype=float)
    for i in range(-n, 0):
        s_t = create_S(x, t + i, w)
        H[:, i+n] = np.transpose(s_t)
    return H

cs = np.zeros(xs.shape, dtype=float)

for i in range(w + n, N-n):
    H1 = create_H(xs, i, w, n)
    H2 = create_H(xs, i + gamma, w, n)

    u1, d1, v1 = np.linalg.svd(H1, full_matrices=False)
    u2, d2, v2 = np.linalg.svd(H2, full_matrices=False)

    max_u = np.transpose(u2[0,:])
    max_v = np.transpose(u1[0,:])

    r = min(n, w)
    U = u1[:r, :]
    acc = 0.0
    for j in range(r):
        acc += np.power(np.dot(max_u, U[j,:]), 2)
        # acc += np.dot(max_u, U[j,:])
    score = 1 - acc
    cs[i] = np.abs(score)

plt.subplot(211)
plt.plot(xs)
plt.subplot(212)
plt.plot(cs)
plt.show()


