# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:45:17 2013
@author: agrigori
"""

import numpy as np
import numpy.random as rnd
from matplotlib import mlab # for percentile calculation

from scipy.spatial import distance as dist
import scipy as sp
import scipy.linalg as la
from scipy.optimize import minimize_scalar # Find optimal lambda for
import itertools
import warnings

# custom imports ->
#from extra_ls_solvers import extra_ls_solvers as ls_solve # to solve regular ELM least square problem
#from utils import data_utils as du # my normalization module
#from utils import numeric_utils as nu
#import mrsr
#from svd_update import svd_update
# custom imports <-

epsilon1 = 1e12
def loo_error(X,Y,loo_type='press',U=None,S=None,lamda=0.0):
    """
    Computes leave-one-out error for system X*beta = Y.
    There are two ways to compute loo_error: regular, by SVD.
    
    SVD method has an advantage that it is easy to evaluate loo error
    for multiple lambda parameters. SVD needed to be computed only once.
    If the SVD method is used then SVD decomposition is computed before the
    function is called with U and S parts of SVD. (X = USVt )
    Input:
        X - matrix X. This can be none of SVD method is used.        
        Y - matrix(or vector) of right hand side
        loo_type - 'press' or 'gcv'
        U - matrix U in SVD (thin SVD is enough)
        S - vector (not a matrix) of singular values as returned by SVD 
            routine.
            If both matrices U and S are provided the SVD method is used,
            otherwise regular methd is used.            
        lamda - regularization parameter    
    Output:
        loo error
    """
    
    if  not loo_type.lower() in ('press', 'gcv'):
        raise ValueError("Function loo_error: parameter loo_type must be either 'press' or 'gcv'. ")
        
    SVD_method = False;
    if ( (not U is None) and (not S is None) ):
        SVD_method = True
    elif (X is None):
        raise ValueError("Function loo_error: For regular method you need to provide X matrix.")
    
    if SVD_method:
        n_samples = U.shape[0]
        
        if (lamda != 0.0):                       
            S = np.power(S,2)
            Sl = S + n_samples*lamda # parameter lambda normalized with respect to number of points !!!
            S = S / Sl
            
            
            (S, orig_S_shape) =  ensure_column(S);
            
            #!
            DUt = np.multiply( S, U.T) # elementwise multiplication, 
            
        else:
            orig_S_shape = None
            DUt = U.T
        
        Mii = np.multiply(U, DUt.T).sum(1) # compute only diagonal of the matrix M        
        D = 1.0/(1.0 - Mii);
        
        MY = np.dot( U ,np.dot(DUt,Y) )
        
    
    else: # regular method        
        (n_samples,X_dim) = X.shape 
        
        XtX = np.dot( X.T, X)
        
        if (lamda != 0.0):   
            XtX = XtX + n_samples*lamda*np.eye(X_dim) # parameter lambda normalized with respect to number of points !!!
            
        chol = sp.linalg.cho_factor(XtX, overwrite_a = True, check_finite=False)
        
        XtX_inv = sp.linalg.cho_solve(chol, np.eye(X_dim) )
        
        M1 = XtX_inv * X.T        
        
        Mii = np.multiply(X, M1.T).sum(1) # compute only diagonal of the matrix M         
        D = 1.0/(1.0 - Mii);
                
        MY = np.dot(X, np.dot( XtX_inv, np.dot(X.T, Y) ) )
    
    
    if loo_type.lower() == 'press':
        (D_col,tmp) = ensure_column(D)
            
        res = np.multiply( D_col, (Y - MY) ) # need vectors are columns in this matrix
        res = np.power(res,2).mean(axis=0) # mean wrt columns 
    else: # GCV
        trace_P = np.sum( 1.0 - Mii)        
                
        res = np.sum(np.power((Y - MY),2), axis=0 ) # sum wrt columns 
        res = (n_samples / trace_P**2) * res       
        
    if orig_S_shape is not None:
        S = rev_ensure_column(S,orig_S_shape)    
        
    return  res  # Sum of LOO MSE for each dimension of Y

#def tmp_ensure_column(dd):
#    old_shape = dd.shape
#    if len(old_shape) == 1:
#        dd.shape = (old_shape[0], 1)
#    else:
#        dd.shape = (old_shape[1], 1) if old_shape[0] == 1
#    
#    return dd, old_shape
   
class Struct(object): # Needed for emulation of structures
    pass    


epsilon2 = 1e12
class LinearModel(object):
    """
    Linear model 
    """

    def __init__(self,**kwargs):
        """
        Constructor, set necessary parameters of the model which are used further.
        
        Input:
            The the dict kwargs these parameters are currently supported.
            
            {'lamda': values} predefined value of regularization parameter
                Note that regularization parameter is multiplied by the number of samples
                in computations.
                
            {'reg_par_optim_type': ('press','gcv','cv','none') } - if lamda is not
                given these optionas are available for searching regularization
                parameter.            
        """
        
        self.model = Struct()
        self.model.orig_kwargs = kwargs        
        
        self.model.lamda = None        
        if 'lamda' in kwargs:
            lamda = kwargs['lamda']
            if lamda > 0:
                self.model.lamda = lamda
        
        self.model.reg_par_optim_type = None
        if ('reg_par_optim_type' in kwargs) and (self.model.lamda is None):
            reg_par_optim_type = kwargs['reg_par_optim_type']
            if (not reg_par_optim_type is None) and (reg_par_optim_type.lower() != 'none'):
                if not reg_par_optim_type in ('press','gcv','cv'):
                    raise ValueError("LinearModel.__init__: unknown values of reg_par_optim_type '%s' " % reg_par_optim_type )
                else:
                    self.model.reg_par_optim_type = reg_par_optim_type                        
                                        
        self.model_built = False
        self.data_set = False
        self.model.type = 'linear'
        
    def __repr__(self):
        """
        Text representation (detailed) of an object.
        """        
        
        return "%s, optim way: %s" % (self.model.type, self.model.reg_par_optim_type )
    
    def set_data(self,X,Y,normalize=True):
        
        """
        Sets data to the model.
        Input:
            X - training features rowwise
            Y - training targets rowwise
            normalize - whether or not normalize training data
         """       
         
        self.data = Struct()
        
        if normalize:
            X_norm,x_means,x_stds = du.normalize(X,ntype=0) # zero mean unit variance    
            Y_norm,y_means,y_stds = du.normalize(Y,ntype=0) # zero mean unit variance                
                
            self.data.normalized = True   
             
            self.data.X = X_norm
            self.data.Y = Y_norm
            
            self.data.x_means = x_means
            self.data.x_stds = x_stds
            
            self.data.y_means = y_means
            self.data.y_stds = y_stds
        else:
            self.data.normalized = False
            
            self.data.X = X
            self.data.Y = Y

        self.model_built = False
        self.data_set = True
        
    def train(self):   
        """
        Training
         
        Input:
           reg_optim_type - type of regularization parameter optimization             
                            Possible values:
                            'none','loo','cv'.
        """
        
        if not self.data_set:
            raise ValueError("LinearModel.train: Data is not set. Training is impossible!")
        
        n_points = self.data.X.shape[0]
        X_d = np.hstack( (np.ones( (n_points ,1)) ,self.data.X) ) # add the column of ones
        Y_d = self.data.Y 
        
        res = self.solve_lin_ols( X_d, Y_d, U=None,S=None,Vh=None )        
        
        self.model.coeffs = res[0]
        self.model.num_rank = res[1]        
        lamda = res[2]
        
        if (lamda is None) or ( lamda > 1.0) or ( lamda < 0.0) : # no lamda optimization happened or wrong optim results
            self.model.lamda = None
        else:
            self.model.lamda = lamda  
                    
        self.model.optim_output = res[3]
        self.model_built = True
             
    def solve_lin_ols(self, X, Y, U=None,S=None,Vh=None):
        """
        Method solves the regularized ols problem, given the SVD decomposition of X.
        Input:
            X - regressor variables
            Y - dependent variables
            U,S,Vh - SVD of X. Actually original X is used only in the cv regularization 
                     method.
                     
        Output:
            res - solution returned by solve_ols_svd.
        """
        #import pdb; pdb.set_trace()
        reg_optim_type = self.model.reg_par_optim_type        
        
        if  not reg_optim_type in (None,'press','gcv','cv'):
            raise ValueError("Linear Model: wrong regularization optimization type %s" % reg_optim_type) 
        
        if U is None: # Perform SVD if it is not done before
            (U,S,Vh) = sp.linalg.svd(X, full_matrices=False, overwrite_a=False, check_finite=False)
       
        n_points = self.data.X.shape[0]
        lamda = None; optim_output = None
        if reg_optim_type in ('press', 'gcv'):
            optim_output = minimize_scalar(lambda lamda: np.sum( loo_error(None,Y, reg_optim_type, U,S,lamda) ), 
                              bounds= (0.0,1.0), method='Bounded')
            lamda = optim_output.x
            
        elif (reg_optim_type == 'cv'):
            
            import sklearn.cross_validation as cv
            
            def cv_optim(par_lamda):
                cv_error = 0
                for cv_train_indeces, cv_test_indeces in cv.KFold(n_points, indices=True, n_folds=10,shuffle = True):
                    X_cv_train = X[cv_train_indeces,:]; Y_cv_train = Y[cv_train_indeces,:]
                    X_cv_test = X[cv_test_indeces,:]; Y_cv_test = Y[cv_test_indeces,:]
                    
                    (U1,S1,Vh1) = sp.linalg.svd(X_cv_train, full_matrices=False, overwrite_a=False, check_finite=False)
                    
                    res = solve_ols_svd(U1,S1,Vh1, Y_cv_train, par_lamda )
                    coeffs = res[0] 
                    
                    Y_predict = np.dot( X_cv_test, coeffs )
                    
                    # compute MSE. This makes sence only if the Y data is normalized 
                    # i.e. different Y dimensions have the same scale.
                    cv_error += np.mean( np.power( (Y_predict - Y_cv_test), 2 ), axis=0 )
         
                return np.sum( cv_error )
         
            optim_output = minimize_scalar(cv_optim, bounds= (0.0,1.0), method='Bounded')
            lamda = optim_output.x
             
        else:  # None             
            if ( S[0]/S[-1] > epsilon2 ): 
                raise ValueError("LinearModel: too large condition number %f and no regularization" % S[0]/S[-1] )
        
        res = solve_ols_svd(U,S,Vh, Y, lamda)
        return res + (lamda,optim_output)
                        
    def predict(self,X_pred,Y_known = None):
        """        
        Predict method.
        
        Input:
            X_pred - data (rowwise) to which prediction to be made
            Y_known - known predictions to compute an error.
            
        """
        
        if not self.model_built:
            raise ValueError("Linear model: Prediction is impossible model is not trained.")
        
        if self.data.normalized:            
            (X_d,tmp1,tmp2) = du.normalize( X_pred, None, self.data.x_means,self.data.x_stds )
        else:
            X_d = X_pred
        
        X_d = np.hstack( ( np.ones( (X_d.shape[0] ,1)) ,X_d ) ) # add the column of ones
        Y_pred = np.dot( X_d, self.model.coeffs)
        if self.data.normalized: 
            Y_pred = du.denormalize( Y_pred, self.data.y_means, self.data.y_stds )
                
        if Y_known is None:
            return (Y_pred, None)
        else:
            return (Y_pred,  np.mean( np.power( Y_pred - Y_known, 2), axis=0  ) )
         
         
    def copy(self):         
        """
            Function which creates a copy of current model, preserving
            the model parameters
        """

        this_class = type(self)     
        new_instance = this_class(**self.model.orig_kwargs)
        
        return new_instance

def solve_ols_svd(U,S,Vh, Y, lamda = 0.0 ):
    """
    Solve OLS problem given the SVD decomposition
    
    Input:
        ! Note X= U*S*Vh and Y are assumed to be normalized, hence lamda is between 0.0 and 1.0.
    
        U, S, Vh - SVD decomposition
        Y - target variables
        lamda - regularization parameter. Lamda must be normalized with respect
                                          to number of samples. Data is assumed
                                          to be normalized, so lamda is between 0.0 and 1.0.
    """
    
    n_points = U.shape[0]
    machine_epsilon = np.finfo(np.float64).eps
    if (lamda is None) or ( lamda > 1.0) or ( lamda < 0.0) : # no lamda optimization happened or wrong optim results
        num_rank = np.count_nonzero(S > S[0] * machine_epsilon)  # numerical rank  
    
        S.shape = (S.shape[0],1)
        coeffs = np.dot( Vh.T, np.multiply( 1.0/S ,np.dot( U.T, Y ) ) )
        
    else:
        S2 = np.power(S,2)
        S2 = S2 + n_points*lamda # parameter lambda normalized with respect to number of points !!!
        S = S / S2
        S.shape = (S.shape[0],1)
        coeffs = np.dot( Vh.T, np.multiply( S ,np.dot( U.T, Y ) ) )
        
        num_rank = None # numerical rank is None because regularization is used
    
    return coeffs, num_rank

def ensure_column(v):
    """
    Function affects only one dimensioanl arrays ( including (1,n) and (n,1) dimensional)
    It then represent the output as a (n,1) vector strictly. It also returns
    initial shape by using which original dimensions of the vector can be reconstructed.
        
    For more than one dimensional vector function do nothing.
        
    Inputs:
        v - array
        
    Output:
        col - the same array with (n,1) dimensions - column
        params - params by which original shape can be restored
    """
    
    initial_shape = v.shape
    if len(initial_shape) == 1: # one dimensional array
        v.shape = (initial_shape[0],1)
    else:
        if (len(initial_shape) == 2)  and (initial_shape[0] == 1): # row vector        
            v.shape = (initial_shape[1],1)     

    return v,initial_shape


def rev_ensure_column(v,initial_shape):
    """
        This function is reverse with respect to ensure_coulmn
        It restores the original dimensions of the vector
        
    """
    if initial_shape: # check that the tuple is nonempty
        v.shape = initial_shape    
    
    return v