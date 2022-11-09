import scipy as sp
import logging

class LinregBatchClient:
    def __init__(self, model, dataset, device, params_dict):
        self._set_training_params(**params_dict.train)
        self._set_data_params(**params_dict.data)
        
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self._prepare_data()
        
    def _set_training_params(self, epochs=None, **kw):
        self.training_params_dict = kw
        
        self.epochs = epochs
        
        self.completed_epochs = 0
        
    def _set_data_params(self, train_split=None, test_split=None, features_key=None, X_dim=None, **kw):
        self.data_params_dict = kw
        
        self.train_split = train_split
        self.test_split = test_split
        self.features_key = features_key
        self.X_dim = X_dim
    
    def _prepare_data(self,):
    
        self.X_train = self.dataset[self.train_split][self.features_key]
        self.X_test = self.dataset[self.test_split][self.features_key]

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
    
    def _get_weigths(self,):
        weights = self.model.get_weights()
        return weights
        
    def predict_train(self):
        pred = self._predict(self.X_train)
            
        return pred
    
    def predict_test(self):
        
        Y_pred = self._predict(self.X_test)
            
        return Y_pred