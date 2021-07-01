import numpy as np
import matplotlib.pyplot as plt

def validate(question,bool_fxn=lambda i,args:i=='y' or i=='n',cast=[str],error_message="Please enter valid input.",*args):
    if not isinstance(cast,tuple):
        cast = [cast]
    ans = input(question + "\n")
    while not bool_fxn(ans, args):
        ans = input(error_message + "\n")
    for i in np.arange(len(cast)):
        try:
            return cast[i](ans)
        except ValueError:
            pass
        
def check_decimal(lr,*args):
    try:
        int(lr)
        return True
    except ValueError:
        pass
    try:
        float(lr)
        return True
    except ValueError:
        pass
    return False
        
def check_posint(n,*args):
    try:
        return int(n) > 0
    except ValueError:
        return False

def get_hyper_params():
    pos_int_error_msg = "Positive integers only. Please try again."
    pos_dec_error_msg = "Positive decimals only. Please try again."
    return validate("# of features?",check_posint,int,pos_int_error_msg),\
        validate("# of samples?",check_posint,int,pos_int_error_msg),\
        validate("max iterations?",check_posint,int,pos_int_error_msg),\
        validate("learning rate?",lambda a,*args:check_decimal(a) and float(a) > 0,float,pos_dec_error_msg)

def init(max_iter,n,d):
    return np.full((max_iter),0.),np.ones((n,1)),np.hstack((np.ones((n,1)),\
        np.random.normal(size=(n,d),scale=1 / np.sqrt(d)))),np.random.normal(size=(n,d + 1)),\
        np.random.normal(size=(n,1)),np.empty((n,d + 1))

def get_losses():
    d,n,max_iter,eta = get_hyper_params()
    losses,v,W,ins,outs,X = init(max_iter,n,d)
    for i in np.arange(max_iter):
        loss = 0
        vgrad = np.zeros((n,1))
        wgrad = np.zeros((n,d + 1))
        for j in np.arange(len(ins)):
            Wx = np.matmul(W,ins[j])
            phiWx = Wx ** 2
            for k in np.arange(n):
                X[k] = ins[j]
            term = np.ndarray.item(outs[j] - np.matmul(v.T,phiWx))
            wgrad = np.add(wgrad,((-2 / n) * term * np.ndarray.item(np.matmul(v.T,Wx))) * X)
            vgrad = np.add(vgrad.reshape(-1,1),((-term / n) * phiWx).reshape(-1,1))
            loss += (term ** 2) / (2 * n)
        v = np.subtract(v,eta * vgrad)
        W = np.subtract(W,eta * wgrad)
        losses[i] = loss
    return losses,n

losses,n = get_losses()
print(losses)
plt.plot(np.linspace(0,len(losses),len(losses)),losses)
plt.show()