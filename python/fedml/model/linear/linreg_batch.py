"""The model class here should contain only parameters of the model, hyperparameters related to the model and
corresponding computational methods. 

Data processing, related methods and parameters should be put into a client/server class.
Returns:

"""

import torch
from . import elm

class LinearRegression_batch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, reg_lambda=0.0, **kw):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            reg_lambda (float): L2 regularizer per 1-sample. That is - regularizer on the sacale of
                MSE error. 
        """
        super(LinearRegression_batch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False, device=None, dtype=None)
        self.reg_lambda = reg_lambda
        self.requires_grad_(False) # turn off gradient computation
        
    def forward(self, x):
        # try:
        outputs = self.linear(x)
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs

    def update_weights(self, data_U, data_S, data_Vh, rhs):
        """Update weights from the SVD of data.

        Args:
            data_U (tensor): U of SVD of data matrix. 
            data_S (tensor): S of SVD of data matrix.
            data_Vh (tensor): Vh of SVD of data matrix.
            rhs (tensor): Y minus prediction of other parties

        Returns:
            coeffs (tensor): new vector of coefficients
        """
        
        coeffs, num_rank = elm.solve_ols_svd(data_U, data_S, data_Vh, rhs, self.reg_lambda)
        
        self.linear.weight.copy_(torch.as_tensor(coeffs).t()) # TODO: copying is not efficient
        
        return coeffs
    
    def predict(self, X_pred):
        
        Y_pred = self.forward(X_pred)
            
        return Y_pred
    
    def get_weights(self,):
        #import pdb; pdb.set_trace()
        weights = self.linear.weight.clone()
        return weights