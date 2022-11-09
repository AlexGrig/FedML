import torch
import scipy as sp
import logging

class LinregBatchServer:
    def __init__(self, model, dataset, device, params_dict):
        self._set_training_params(**params_dict.train)
        self._set_data_params(**params_dict.data)
        
        self.model = model
        self.dataset = dataset
        self.device = device
        self.pad_ones = torch.nn.ConstantPad1d((1,0), 1.0)
        self._prepare_data()
        
    def _set_training_params(self, epochs=None, **kw):
        self.training_params_dict = kw
        
        self.epochs = epochs
        
        self.completed_epochs = 0
        
    def _set_data_params(self, train_split=None, test_split=None, features_key=None, label_key=None, X_dim=None, Y_dim=None, **kw):
        self.data_params_dict = kw
        
        self.train_split = train_split
        self.test_split = test_split
        self.features_key = features_key
        self.label_key = label_key
        
        self.X_dim = X_dim
        self.Y_dim = Y_dim
    
    def _prepare_data(self,):
        #import pdb; pdb.set_trace()
        
        self.X_train = self.pad_ones( self.dataset[self.train_split][self.features_key] )
        self.X_test = self.pad_ones( self.dataset[self.test_split][self.features_key] )
        
        self.Y_train = torch.atleast_2d( self.dataset[self.train_split][self.label_key] ).t()
        self.Y_test =  torch.atleast_2d( self.dataset[self.test_split][self.label_key] ).t()
        
        (self.U, self.S, self.Vh) = sp.linalg.svd(self.X_train.numpy(), full_matrices=False, overwrite_a=False, check_finite=False)
        
    def _predict(self, X_pred):
        #import pdb;pdb.set_trace()
        pred = self.model.predict(X_pred)
        return pred
    
    def update_model_weights(self, righthand_side):
        #import pdb;pdb.set_trace()
        #if self.completed_epochs < self.epochs:
        self.model.update_weights(self.U,self.S,self.Vh, righthand_side.numpy())
        self.completed_epochs += 1
        #else:
        #    pass
        
    def predict_train(self, return_residuals=False):
        pred = self._predict(self.X_train)
        
        if return_residuals:
            ret = pred - self.Y_train
        else:
            ret = pred
            
        return ret
    
    def predict_test(self, return_residuals=False):
        
        Y_pred = self._predict(self.X_test)
        
        if return_residuals:
            ret =  Y_pred - self.Y_test
        else:
            ret = Y_pred
            
        return ret

    def get_Y_train(self,):
        return self.Y_train
    
    def get_Y_test(self,):
        return self.Y_test