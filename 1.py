import numpy as np
import matplotlib.pyplot as plt


eta = .01
x0 = np.zeros((100,1))

ind = np.random.choice(np.arange(0,len(x0)), 20)

x0[ind] = np.random.normal(0,1,np.shape(ind)).reshape(-1,1)

plt.stem(x0)
plt.show()

N = len(x0)
import numpy as np
pi=np.pi


import numpy as np
def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

measurements = np.matmul(DFT_matrix(len(x0)),(x0.reshape(-1,1)))

idx = np.random.choice(len(x0), size = 30, replace = False)
A = DFT_matrix(len(x0))[idx,:]

w, v = np.linalg.eig(np.matmul(A, np.conjugate(A).T))
b = measurements[idx]
print(b.shape)
iterations = 1000
epsilon = np.full((100,1),.0001)
xvector = np.ones((100,1))
norms = np.array([.5,1,2])
ah_times_aahinverse = np.matmul(A.T.conjugate(),np.linalg.inv(np.matmul(A,A.T.conjugate())))
losses = np.full((len(norms),iterations),0,dtype=np.complex128)
for i in np.arange(len(norms)):
    for j in np.arange(iterations):
        xvector = np.subtract(xvector,eta * np.dot(np.power(np.power(np.power(xvector,2) + epsilon,1 / 2),norms[i] - 2).reshape(1,-1),xvector))
        stdloss = np.subtract(b,np.matmul(A,xvector))
        xvector = np.add(xvector,np.matmul(ah_times_aahinverse,stdloss))
        losses[i][j] += (np.sum(np.power(xvector,norms[i])) + np.dot(stdloss.reshape(1,-1),stdloss)).imag
fig, axs = plt.subplots(len(norms))
xaxis = np.linspace(1,iterations,iterations)
for i in np.arange(len(norms)):
    axs[i].plot(xaxis,losses[i])
    axs[i].set_title("L" + str(norms[i]) + " norm")
plt.show()