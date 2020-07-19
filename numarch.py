import numpy as np
import math
import warnings
import time
from scipy.optimize import minimize
from helpers import *

class arch:
    def __init__(self, model='bekk-arch'):
        # Create variables
        if model not in ['bekk-arch', 'bekk-garch']:
            raise Exception('Unknown model \'{}\''.format(model))
        self.model = model # Model type
        self.estimates = None # Parameter estimates
        self.cov = None # Sandwich formula estimate of covariance matrix
        self.T = None # Number of observations
        self.p = None # Number of variables
        self.q = None # Number of elements above diagonal in pxp matrix
        self.d = None # Number of scalar parameters
        self.X = None # Data
        self.Sxx = None # Sample (unconditional) covariance matrix
        self.scaling = 1 # Scaling for log-likelihood fu nction
        
    # Negative log likelihood
    def nllik(self, theta):
        '''Compute negative log-likelihood in parameter vector theta'''
        if self.model == 'bekk-arch' or self.model =='bekk-garch':
            # Calculate log-likelihood
            nllik = np.sum(self.lik_con(theta))/self.scaling
            return -nllik
        
    def lik_con(self, theta):
        '''Compute log-likelihood contributions in parameter vecor theta'''
        param_dict = self.unpacker(theta)
        if self.model == 'bekk-arch':
            # Get input
            Omega  = param_dict['Omega']
            A      = param_dict['A']

            # Compute Omega_t's
            OMEGA_T = [Omega + A @ self.X[t,].reshape(self.p,1)
                                 @ self.X[t,].reshape(1,self.p) @ A.T
                       for t in range(self.T)]

            # Return log-likelihood contributions
            return [-0.5*(self.p*np.log(2*math.pi)
                          + np.log(np.linalg.det(OMEGA_T[t]))
                          + self.X[t+1,].reshape(1,self.p)
                          @ np.linalg.inv(OMEGA_T[t])
                          @ self.X[t+1,].reshape(self.p,1))
                    for t in range(self.T-1)]
        elif self.model == 'bekk-garch':
            # Get input
            Omega  = param_dict['Omega']
            A      = param_dict['A']
            B      = param_dict['B']
            
            # Compute conditional covariance matrices
            OMEGA_T = []
            for t in range(self.T):
                if t > 0:
                    Omega_t = Omega + A @ self.X[t-1,].reshape(self.p,1)\
                                        @ self.X[t-1,].reshape(1,self.p) @ A.T\
                                    + B @ OMEGA_T[-1] @ B.T
                else:
                    Omega_t = self.Sxx
                OMEGA_T.append(Omega_t)
                
            # Return log-likelihood contributions
            return [-0.5*(self.p*np.log(2*math.pi)
                          + np.log(np.linalg.det(OMEGA_T[t]))
                          + self.X[t+1,].reshape(1,self.p)
                          @ np.linalg.inv(OMEGA_T[t])
                          @ self.X[t+1,].reshape(self.p,1))
                    for t in range(self.T-1)]

    # Constraint
    def cons(self, theta):
        '''Constrain the parameter space of the maximization problem'''
        if self.model == 'bekk-arch' or self.model == 'bekk-garch':
            # Get Omega
            Omega = self.unpacker(theta)['Omega']
            return pos_def(Omega)
    
    # Unpack theta into estimates
    def unpacker(self, theta):
        '''Unpack parameter vector theta into a more readable format'''
        if self.model == 'bekk-arch':
            return {'Omega': unvech(theta[:self.q]),
                    'A': theta[self.q:].reshape(self.p,self.p)}
        elif self.model == 'bekk-garch':
            return {'Omega': unvech(theta[:self.q]),
                    'A': theta[self.q:self.q+self.p**2].reshape(self.p,self.p),
                    'B': theta[self.q+self.p**2:].reshape(self.p,self.p)}
        
    # Initial values
    def init_theta(self):
        '''Initialize the parameter vector'''
        if self.model == 'bekk-arch':
            Omega_0 = vech(self.Sxx)
            A_0     = 0.5*np.eye(self.p).flatten()
            theta_0 = np.concatenate([Omega_0, A_0])
            self.d  = len(theta_0)
            return theta_0
        elif self.model == 'bekk-garch':
            Omega_0 = vech(self.Sxx)
            A_0     = 0.5*np.eye(self.p).flatten()
            B_0     = 0.5*np.eye(self.p).flatten()
            theta_0 = np.concatenate([Omega_0, A_0, B_0])
            self.d  = len(theta_0)
            return theta_0
        
    # Check model staionarity
    def stationarity(self, theta):
        '''Check if model with theta is covariance stationarity'''
        param_dict = self.unpacker(theta)
        if self.model == 'bekk-arch':
            Omega = param_dict['Omega']
            A = param_dict['A']
            try:
                rho = sorted(np.abs(np.linalg.eig(
                             np.kron(A,A))[0]), reverse=True)[0]
                if rho < 1:
                    print('\nCondition for covariance stationarity '
                          'confirmed: {:.2f} < 1'.format(rho))
                else:
                    print('\nCondition for covariance stationarity '
                          'failed: {:.2f} > 1'.format(rho))
            except:
                rho = A**2
                if rho < 1:
                    print('\nCondition for covariance stationarity '
                          'confirmed: {:.2f} < 1'.format(rho))
                else:
                    print('\nCondition for covariance stationarity '
                          'failed: {:.2f} > 1'.format(rho))
        elif self.model == 'bekk-garch':
            Omega = param_dict['Omega']
            A = param_dict['A']
            B = param_dict['B']
            try:
                rho = sorted(np.abs(np.linalg.eig(
                             np.kron(A,A)+np.kron(B,B))[0]), reverse=True)[0]
                if rho < 1:
                    print('\nCondition for covariance stationarity '
                          'confirmed: {:.2f} < 1'.format(rho))
                else:
                    print('\nCondition for covariance stationarity '
                          'failed: {:.2f} > 1'.format(rho))
            except:
                rho = A**2 + B**2
                if rho < 1:
                    print('\nCondition for covariance stationarity '
                          'confirmed: {:.2f} < 1'.format(rho))
                else:
                    print('\nCondition for covariance stationarity '
                          'failed: {:.2f} > 1'.format(rho))
        
    # Fit model        
    def fit(self, X, se = True, options = {'disp': False, 'ftol': 1e-06}):
        '''Fit model
        
        Keyword arguments:
        X       -- Data (T x p array)
        se      -- Boolean indicating if standard errors should be computed
        options -- Options for scipy.optimize.minimize
        '''
        print('BEGAN ESTIMATING {}...'.format(self.model.upper()))
        
        # Get data
        self.X = X
        
        # Calculate covariance matrix
        self.T = self.X.shape[0]
        self.Sxx = np.dot(self.X.T, self.X)/self.T
        
        # System dimensionality
        self.p = self.X.shape[1]
        self.q = int(self.p*(self.p+1)/2)
        
        # Initialize theta and find scaling
        theta_0 = self.init_theta()
        f_theta_0 = self.nllik(theta_0)
        self.scaling = abs(f_theta_0/2)
        
        # Print some output
        print('Number of observations (T): ', self.T)
        print('Number of variables (p): ', self.p)
        
        # Minimize
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        t0 = time.time()
        res = minimize(self.nllik, theta_0, method='SLSQP',
                       constraints = {'type': 'ineq', 'fun': self.cons},
                       options = options)
        t1 = time.time()
        warnings.resetwarnings()
        if res.success:
            print('\nNumerical optimization succesfully converged '
                  'in {:.2f} seconds'.format(t1-t0))
        else:
            print('\nWARNING: Numerical optimization did not converge')
        
        # Calculate MLE
        self.estimates = self.unpacker(res.x)
        print('\nParameter estimates:')
        print(self.estimates)
        
        # Calculate standard errors
        if se:
            s = num_derivs(self.lik_con, res.x).reshape(self.T-1, self.d)
            B = (1/(self.T-1)) * s.T @ s
            def jac(x): return list(num_derivs(self.nllik,x))
            hes = num_derivs(jac, res.x)
            A_inv = np.linalg.inv(hes)
            self.cov = (1/(self.T-1)) * A_inv @ B @ A_inv
            theta_se = np.sqrt(np.diag(self.cov))
            self.std_errors = self.unpacker(theta_se)
            print('\nStandard errors (sandwich formula):')
            print(self.std_errors)
            
        # Check stationarity
        self.stationarity(res.x)