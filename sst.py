import numpy as np
import matplotlib.pyplot as plt

N = 5000
ts = np.linspace(0, 5, N)

# N1 = 200
# N2 = 200
# N3 = 200
# N4 = 200
# N5 = 200
# xs1 = np.sin(100 * np.pi * ts[:N1])
# xs2 = np.sin(20 * np.pi * ts[N1:N2+N1])
# xs3 = np.cos(100 * np.pi * ts[N2+N1:N3+N2+N1])
# xs4 = np.sin(20 * np.pi * ts[N3+N2+N1:N4+N3+N2+N1])
# xs5 = np.sin(100 * np.pi * ts[N4+N3+N2+N1:N5+N4+N3+N2+N1])
# xs = np.hstack([xs1, xs2, xs3, xs4, xs5])

# rs = np.random.random(N)
# xs = xs + rs * 0.05

xs1 = ts[:2500]
xs2 = np.ones((2500)) * ts[2500]

xs = np.hstack([xs1, xs2])
print xs.shape

# Change-Point Detection of Climate Time Series by Nonparametric Method
# http://www.iaeng.org/publication/WCECS2010/WCECS2010_pp445-448.pdf
# n = w, gamma = n/2
N = xs.shape[0]


w = 10
n = 10
gamma = 5

def create_S(x, t, w):
    return x[t-w:t]

def create_H(x, t, w, n):
    H = np.zeros((w, n), dtype=float)
    print t - n -w, t - 1
    for i in range(-n, 0):
        s_t = create_S(x, t + i, w)
        H[:, i+n] = np.transpose(s_t)
    return H

# i = 1500
# H1 = create_H(xs, i, w, n)
# H2 = create_H(xs, i + gamma, w, n)
#
# print H1
# print H2
#
# u1, d1, v1 = np.linalg.svd(H1, full_matrices=False)
# u2, d2, v2 = np.linalg.svd(H2, full_matrices=False)
# print "u1", u1
# print "u2", u2
#
# max_u = u2[:,0]
# max_v = u1[:,0]
#
#
# print np.dot(max_u, max_v)
# print 1 - np.dot(max_u, max_v) ** 2
#
# assert False

cs = np.zeros(xs.shape, dtype=float)
offset = gamma + np.max([w, n]) + 1
for i in range(offset + 500, N - offset-100):
# for i in range(offset, offset + 2):
# for i in range(w + n, w + n + 50):
    i += 20
    H1 = create_H(xs, i, w, n)
    H2 = create_H(xs, i + gamma, w, n)

    u1, d1, v1 = np.linalg.svd(H1, full_matrices=False)
    u2, d2, v2 = np.linalg.svd(H2, full_matrices=False)

    max_u = np.transpose(u2[:,0])
    max_v = np.transpose(u1[:,0])


    r = min(n, w)
    U = u1[:, :r]
    acc = 0.0
    for j in range(U.shape[1]):
        b = np.dot(U[:,j], max_u)
        acc += b**2
        # acc += np.dot(max_u, U[j,:])
    score = 1 - acc
    cs[i] = np.abs(score)
    print score
    # cs[i] = np.arccos(np.dot(max_u, max_v)) / np.pi

plt.subplot(211)
plt.plot(xs)
plt.subplot(212)
plt.plot(cs)
plt.show()


