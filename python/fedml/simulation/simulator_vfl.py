import importlib
import datasets
simulator_mpi_algorithm_path = 'fedml.simulation.mpi'

class SimulatorMPI:
    simulator_mpi_algorithm_path = 'fedml.simulation.mpi'
    
    @staticmethod
    def _validate_params(params):
        
        error_prefix = 'SimulatorMPI params valiadation: '
        
        # temp: torn off for developing
        #assert params.common.parties_num == params.data.data_partition_parts, \
        #    error_prefix + f'"parties_num" ({params.common.parties_num}) and "data_partition_parts" ({params.data.data_partition_parts}) must be equal.'
        
        assert (params.common.role == 'server' and params.data.with_labels == True) or \
            (params.common.role == 'client' and params.data.with_labels == False), error_prefix + 'server must have labels ans client must not.'
        
            
    def __init__(self, params, device, dataset, model):
    
        common_params_dict = params.common
        train_params_dict = params.train
        data_params_dict = params.data
        
        self._validate_params(params)
        
        #import pdb; pdb.set_trace()
        if isinstance(model, dict):
            model = model( list(model.keys())[0] )
        if isinstance(dataset, dict) and not isinstance(dataset, datasets.DatasetDict):
            dataset = dataset[ list(dataset.keys())[0] ]

        algorithm_name = common_params_dict.algorithm
        
        algorithm_module = importlib.import_module(f'{self.simulator_mpi_algorithm_path}.{algorithm_name}')
        if common_params_dict.role == 'server':
            server =  algorithm_module.server_class(model, dataset, device, params) # Instantiate  server/client model
            server_manager = algorithm_module.server_manager_class(params.connection, params.train + params.validation, server, params.common.backend.upper())
            server_manager.run()
        elif common_params_dict.role == 'client':
            client =  algorithm_module.client_class(model, dataset, device, params) # Instantiate  server/client model
            client_manager = algorithm_module.client_manager_class(params.connection, params.train, client, params.common.backend.upper())
            client_manager.run()
        else:
            raise ValueError("Role must be either server or client!")