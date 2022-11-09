import torch
class Sum(torch.nn.Module):
    def __init__(self, client_model):
        super(Sum, self).__init__()
        self.client_model=client_model

    def forward(self, x):
        # try:
        outputs = torch.sum(x)
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
        
        coeffs = self.client_model.update_weights(data_U, data_S, data_Vh, rhs)
        
        return coeffs
    
    def predict(self, X_pred):
        client_model_predict = self.client_model.predict(X_pred)
        
        return client_model_predict
    
    def get_weights(self,):
        weights = self.client_model.get_weights()
        
        return weights
    
    