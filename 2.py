from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
indices01 = (y.to_numpy() == '1') | (y.to_numpy() == '0')
only01 = X.to_numpy()[indices01] / 255
training = np.insert(only01,0,np.ones(len(only01)),axis=1)
test = y.to_numpy()[indices01].astype('float64')

def sigmoid(x,weights,label):
    x = x.reshape(-1,1)
    logistic = 1 / (1 + np.exp(-np.matmul(weights.reshape(1,-1),x)))
    grad = None
    if label == 1:
        grad = (logistic - 1) * x
    else:
        grad = logistic * x
    return logistic,grad


epsilon = .001


def logloss(x,weights,label):
    logistic,grad = sigmoid(x,weights,label)
    logloss = None
    if label == 1:
        logloss = -np.log(logistic + epsilon)
    else:
        logloss = -np.log(1 - logistic + epsilon)
    return logloss,grad  


epochs = 100
loglosses = np.empty(epochs)
eta = .0001
weights = np.full((len(training[0]),1),.001)
for i in np.arange(epochs):
    loglossepoch = 0
    print(i)
    for j in np.arange(len(training)):
        ll,grad = logloss(training[j],weights,test[j])
        weights = np.subtract(weights,eta * grad)
    for j in np.arange(len(training)):
        ll,grad = logloss(training[j],weights,test[j])
        loglossepoch += ll
        loglosses[i] = loglossepoch
plt.plot(np.linspace(1,epochs,epochs),loglosses)
plt.show()