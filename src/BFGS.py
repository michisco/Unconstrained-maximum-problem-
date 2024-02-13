import pandas as pd
import numpy as np
import time

import exceptionHandler as exc
import configurations as config

class BFGS():
  def __init__(self, obj_function, *args, verbose = False):
    '''Arguments:
      - obj_function: objective function
      - x: initial point (optional)
      - beta (optional)
      - epsilon: stopping criteria norm of gradient (optional)
      - fstar: stopping criteria with optimal f (optional)
      - verbose: if true print the results
    '''
    
    self.f = obj_function
    if len(args) == 0:
      self.x = np.random.rand(self.f.n, 1)
    else:
      self.x = args[0]
      if not np.all(np.isreal(self.x)):
          raise exc.ErrorExc( 'x is not a real vector' )

      if self.x.shape[1] != 1:
          raise exc.ErrorExc( 'x is not a (column) vector' )

    self.n = self.x.shape[0]

    if len(args) > 1:
      self.beta = args[1]
      if not np.isscalar( self.beta ):
          raise exc.ErrorExc( 'beta is not a real scalar' )
    else:
      self.beta = 1
      
    if len(args) > 2:
      self.eps = args[2]
      if not np.isreal( self.eps ) or not np.isscalar( self.eps ):
          raise exc.ErrorExc( 'eps is not a real scalar' )
    else:
      self.eps = config.eps
      if not np.isreal( self.eps ) or not np.isscalar( self.eps ):
          raise exc.ErrorExc( 'eps is not a real scalar' )
          
    if len(args) > 3:
      self.fstar = args[3]
      if not np.isreal( self.fstar ) or not np.isscalar( self.fstar ):
          raise exc.ErroExc( 'fstar is not a real scalar' )
    else:
        self.fstar = config.mInf
        
    self.verbose = verbose

    self.maxIter = round(config.maxIter)
    if not np.isscalar(self.maxIter):
        raise exc.ErrorExc( 'maxIter is not an integer scalar' )

    self.mInf = config.mInf
    if not np.isscalar( self.mInf ):
        raise exc.ErrorExc( 'MInf is not a real scalar' )

    self.minA = config.minStep
    if not np.isscalar(  self.minA ):
       raise exc.ErrorExc( 'minA is not a real scalar' )
    if self.minA < 0:
       raise exc.ErrorExc( 'minA is < 0' )      

    self.last_x = np.zeros((self.n , 1))  # last point visited
    self.last_g = np.zeros((self.n , 1))  # gradient of last_x
    self.d = np.zeros((self.n , 1))    # quasi-Newton's direction
    
    self.H = self.beta * np.eye(self.n) #starting Hessian matrix H0
    self.feval = 1 #counter of function evaluations
    
    self.last_x = self.x
    self.last_v, self.last_g = self.f.compute(self.last_x) #calculating f(x+alpha*d) with d = 0

    self.g = self.last_g
    self.v = self.last_v
    self.ng = np.matmul(self.g.T, self.g)
    self.ng = np.sqrt(self.ng)
    if config.eps < 0:
      self.ng0 = -self.ng
    else:
      self.ng0 = 1

  def applyBFGS(self):
    #storing the norms and values for each steps
    self.computed_norms = []
    self.computed_fx =  []
    
    while True:
      if self.verbose:
        self.printVerbose()
        
      self.computed_fx.append(self.v.item())
      self.computed_norms.append(self.ng.item())
        
      #if fstar is not equal to -inf we evaluate the gap rather than the norm of gradient
      if self.fstar != config.mInf:
          err = abs(self.v - self.fstar) / abs(self.fstar)
          
          if err <= 1e-15:
              self.status = 'optimal'
              return self.computed_norms, self.computed_fx, self.status
      else:
          #if the norm of the gradient is lower or equal of the epsilon then stop
          if self.ng <= self.eps * self.ng0:
              self.status = 'optimal'
              return self.computed_norms, self.computed_fx, self.status
          
      #if we reach the maximum number of iteration then stop
      if self.feval > config.maxIter:
          self.status = 'stopped'
          return self.computed_norms, self.computed_fx, self.status

      #compute approximation to Newton's direction
      self.d = np.matmul(-self.H, self.g)
            
      #calculate exact line search
      self.alpha = self.f.exactLS(self.d)
      
      self.last_x = self.x + self.alpha * self.d 
      self.last_v, self.last_g = self.f.compute(self.last_x)
      self.feval = self.feval + 1
      
      if self.alpha <= self.minA:
        self.status = 'error min step'
        return self.computed_norms, self.computed_fx, self.status
      
      if self.v <= self.mInf:
        self.status = 'unbounded'
        return self.computed_norms, self.computed_fx,self.status
      
      #update approximation of the Hessian computing s and y
      s = self.last_x - self.x   #s_i = x_{i + 1} - x_i
      y = self.last_g - self.g   #y_i = \nablaf( x_{i + 1} ) - \nablaf( x_i )

      rho = np.matmul(y.T, s)
      
      #check if rho is not negative
      if rho < 1e-16:
        self.status = 'error rho'
        return self.computed_norms, self.computed_fx, self.status
          
      rho = 1 / rho
      
      ''' BFGS update
      H_{k+1} = H_k + ρ_k*((1 + ρ_k*y_k'*H_k*y_k)s_k*s_k' - (H_k*y_k*s_k' + s_k*y_k'*H_k))
      where ρ_k = 1 / (y_k^T * s_k) 
      '''
      temp_H1 = np.matmul(self.H, y)
      temp_H1 = np.matmul(temp_H1, s.T)
      temp_H2 = np.matmul(y.T, self.H)
      temp_H2 = np.matmul(temp_H2, y)
      self.H = self.H + rho * ( ( 1 + rho * temp_H2) * np.matmul( s, s.T) - temp_H1 - temp_H1.T )

      #update variables for next iteration
      self.x = self.last_x
      self.v = self.last_v
      self.g = self.last_g
      self.ng = np.matmul(self.g.T, self.g)
      self.ng = np.sqrt(self.ng)
    
  def printVerbose(self):
        print("fval: %d f(x): %0.15f ||g||: %0.15f" %(self.feval, self.v, self.ng))