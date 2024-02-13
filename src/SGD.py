import numpy as np
import math
from numpy import linalg as la
import configurations as config
import objectiveFunction as of
import time 

class SGD():
    def __init__(self, obj_function, *args, verbose = False):
        '''Arguments:
          - obj_function: objective function
          - x: initial point (optional)
          - eps: stopping criteria norm of gradient (optional)
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
        
        if len(args) > 1:
          self.eps = args[1]
          if not np.isreal( self.eps ) or not np.isscalar( self.eps ):
              raise exc.ErrorExc( 'eps is not a real scalar' )
        else:
          self.eps = config.eps
          if not np.isreal( self.eps ) or not np.isscalar( self.eps ):
              raise exc.ErrorExc( 'eps is not a real scalar' )
        
        if len(args) > 2:
          self.fstar = args[2]
          if not np.isreal( self.fstar ) or not np.isscalar( self.fstar ):
              raise exc.ErroExc( 'fstar is not a real scalar' )
        else:
          self.fstar = config.mInf
    
        self.n = self.x.shape[0]
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
           
        self.fx, self.gx = self.f.compute(self.x)
        self.feval = 1
        self.status = ''
        self.ng = np.linalg.norm(self.gx)
    
    
    def applySGD(self):
        x_new = np.zeros(self.n)
        g_new = np.zeros(self.n)
        
        if self.verbose:
            self.printVerbose()
            
        # Prepare list to store results at each iteration 
        self.results = []
        self.gradient_norms = []
            
        # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        while True:
            self.results.append(self.fx.item())
            self.gradient_norms.append(self.ng.item())
            
            #if fstar is not equal to -inf we evaluate the gap rather than the norm of gradient
            if self.fstar != config.mInf:
              err = abs(self.fx - self.fstar) / abs(self.fstar)
          
              if err <= 1e-15:
                self.status = 'optimal'
                break
            else:
              #if the norm of the gradient is lower or equal of the epsilon then stop
              if self.ng <= self.eps:
                self.status = 'optimal'
                break

            if self.feval > self.maxIter:
                self.status = 'stopped, maxIter'
                break
            
            # Compute step size with exact line search
            a = self.f.exactLS(-self.gx)
            
            self.x = self.x - a * self.gx
            self.fx, self.gx = self.f.compute(self.x)
            # compute and update the norm of gradient
            self.ng = np.linalg.norm(self.gx)
            self.feval += 1
            
            if self.verbose:
                self.printVerbose()
            
            # Output statistics
            if a <= self.minA:
                self.status = 'error minStep'
                break

            if self.fx <= self.mInf:
                self.status = 'unbounded'
                break

        return self.results, self.gradient_norms, self.status
    
    def printVerbose(self):
        print("fval: %d f(x): %0.15f ||g||: %0.15f" %(self.feval, self.fx, self.ng))