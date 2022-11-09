import fedml
from fedml import FedMLRunner
from fedml import AttrDict
import yaml
import json
import logging

from fedml.data.vert_fl_dataloader import load, DataPreprocessor, update_params

def make_server():
    pass

def pretty_print_params(d, indent=0):
   for key, value in d.items():
      if isinstance(value, dict):
         print('\t' * indent + f'{key}:', flush=True)
         pretty_print_params(value, indent+1)
      else:
         print('\t' * indent + f'{key}: {value}', flush=True)
         
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # init FedML framework
    params = fedml.init2()

    # init device
    # import pdb; pdb.set_trace()
    device = fedml.device.get_device(params.common + params.device)
    #import pdb; pdb.set_trace()
    # load data
    
    # Load data and update data params ->
    dataset, updated_data_params = load(params.common + params.data)
    params = update_params(params, updated_data_params, 'data')
    # Load data and update data params <-
    
    # Preprocess data and update data and commnon params ->
    dp = DataPreprocessor(dataset, params.data)
    dataset, updated_data_params = dp.preprocess()
    params = update_params(params, updated_data_params, 'data')
    params = update_params(params, updated_data_params, 'model')
    params.common.with_labels = updated_data_params['with_labels'] # update common params with with_labels.
    # Preprocess data and update data and commnon params <-
    
    #import pdb; pdb.set_trace()
    
    # create model ->
    model = fedml.model.create2(params.model)
    # create model ->
    
    # start training
    params['training_type'] = params.common['training_type']
    params['backend'] = params.common['backend']
    
    #print(model)
    #print(params)
    pretty_print_params(params)
    
    #import pdb; pdb.set_trace()
    fedml_runner = FedMLRunner(params, device, dataset, model)
    #fedml_runner.run()
    
    
    
    #Comments:
    # 1) Regarding model classes:    
        # model - parameters, and necessary math algorithms. 
        # model_client/model_server - optimizer, data, training procedures.
        # client_manager/server_maneger - messages to exchange with. Training algorithm: in term of messages exchange
        
    # 2) Regarding config and parameters:
        # a) Config/parameters are divided onto predefined sections
        # b) Each section is responsible for its own significant component
        # c) There is also a common section.
        # d) During initialization the sections are updated based on other sections.
            # 1) E.g. model section is updated based on loaded data.
            # 2) Parameters can be added but can not be changed.
        # e) 