import numpy as np

class objectiveFunction():
    def __init__(self, A):
        self.A = A
        self.Q = np.matmul(np.transpose(self.A), self.A)
        self.n = self.Q.shape[0]
    
    #compute and returns fx and its gradient
    def compute(self, x):
        self.x = x
        self.xT = x.T
        self.Qx = np.matmul(self.Q,x)
        #numerator = x'Qx
        self.xTQx = np.matmul(self.xT, self.Qx)
        #denominator = x'x
        self.xTx = np.matmul(self.xT, x)
        #f(x) = x'Qx / x'x
        fx = self.xTQx / self.xTx
        #gradf(x) = 2xf(x) / x'x - 2Qx / x'x
        grad_fx = ((2 * x * fx) / self.xTx) - ((2 * self.Qx) / self.xTx)
        return fx, grad_fx
       
    #compute the exact line search and returns the step size  
    def exactLS(self, d):
        alpha = 0
        Qd = np.matmul(self.Q, d)
        dTQd = np.matmul(d.T, Qd)
        xTQd = np.matmul(self.xT, Qd)
        dTd = np.matmul(d.T, d)
        xTd = np.matmul(self.xT, d)
        
        #a = (d'Qd)(x'd) - (x'Qd)(d'd) 
        a = float((dTQd * xTd) - (xTQd * dTd))
        # b = (d'Qd)(x'x) - (x'Qx)(d'd)
        b = float((dTQd * self.xTx) - (self.xTQx * dTd))
        # c = (x'Qd)(x'x) - (x'Qx)(x'd)
        c = float((xTQd * self.xTx) - (self.xTQx * xTd))
        
        # finding the solution on a*alpha^2 + b*alpha + c with alpha > 0 
        coeff_array  = np.array([a, b, c])
        res = np.roots(coeff_array)
        
        #check if the results are > 0
        if res[0] < 0 and res[1] < 0:
            alpha = 0
        elif res[0] < 0:
            alpha = res[1]
        elif res[1] < 0:
            alpha = res[0]
        else:
            alpha = np.min([res])
        
        #in case returns the minimum solutions from polynomial     
        return alpha