import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class ComputeAUC():
    def __init__(self, name = 'AUC', sigmoid_transform=True):
        self.name = name
        self.sigmoid_transform = sigmoid_transform
    
    def compute(self, true_label, predictions):
        predictions = predictions.detach().clone().squeeze() #.clone()
        true_label = true_label.detach().clone().squeeze()
        
        if self.sigmoid_transform:
            predictions = torch.sigmoid(predictions)
        auc = roc_auc_score(true_label.numpy(), predictions.numpy())
        return auc

class ComputeAccuracy():
    def __init__(self, name = 'Accuracy', positive_int = 1, negative_int=-1, force_to_binary=True):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to 'Accuracy'.
            positive_int (int, optional): Depends on the ture labels. Defaults to 1.
            negative_int (int, optional): Depends on the ture labels. Defaults to -1.
            force_to_binary (bool, optional): _description_. Defaults to True.
        """
        self.force_to_binary = force_to_binary
        self.name = name
        self.positive_int = positive_int
        self.negative_int = negative_int
        
    def compute(self, true_label, predictions):
        predictions = predictions.detach().clone().squeeze()
        true_label = true_label.detach().clone().squeeze()
        
        if self.force_to_binary:
            predictions[predictions<0] = self.negative_int; predictions[predictions>0] = self.positive_int
        #import pdb; pdb.set_trace()
        #accuracy = accuracy_score(true_label.numpy(), predictions.numpy())
        accuracy = torch.sum(true_label == predictions)/true_label.shape[0]
        return float(accuracy)

class ComputeRMSE():
    def __init__(self, name='RMSE'):
        self.name = name
    
    def compute(self, true_label, predictions):
        predictions = predictions.detach().clone().squeeze()
        true_label = true_label.detach().clone().squeeze()
        
        #rmse = np.sqrt( np.sum((true_label - predictions)**2)/true_label.shape[0] )
        rmse = torch.sqrt( torch.pow(true_label - predictions, 2).mean(dim=0) )
        return float(rmse)

def sigmoid_numpy(arr):
    #import pdb; pdb.set_trace()
    res = 1.0 / (1.0 + np.exp(-arr))
    return res

class ComputeAUC_numpy():
    def __init__(self, sigmoid_transform=True):
        self.name = 'AUC'
        self.sigmoid_transform = sigmoid_transform
    
    def compute(self, true_label, predictions):
        predictions = predictions.copy()
        true_label = true_label.copy()
        
        if self.sigmoid_transform:
            predictions = sigmoid_numpy(predictions)
        auc = roc_auc_score(true_label.squeeze(), predictions.squeeze())
        return auc

class ComputeAccuracy_numpy():
    def __init__(self, force_to_binary=True):
        self.force_to_binary = force_to_binary
        self.name = 'Accuracy'
        
    def compute(self, true_label, predictions):
        predictions = predictions.copy()
        true_label = true_label.copy()
        
        if self.force_to_binary:
            predictions[predictions<0] = -1; predictions[predictions>0] = 1
        #import pdb; pdb.set_trace()
        #accuracy = accuracy_score(true_label.squeeze(), predictions.squeeze())
        accuracy = np.sum(true_label.squeeze() == predictions.squeeze())/true_label.squeeze().shape[0]
        return accuracy
    
class ComputeRMSE_numpy():
    def __init__(self):
        self.name = 'RMSE'
    
    def compute(self, true_label, predictions):
        predictions = predictions.copy()
        true_label = true_label.copy()
        
        rmse = np.sqrt( np.sum((true_label - predictions)**2)/true_label.shape[0] )
        return rmse