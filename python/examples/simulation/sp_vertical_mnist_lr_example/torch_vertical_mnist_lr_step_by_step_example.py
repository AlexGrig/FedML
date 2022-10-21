import fedml
from fedml import FedMLRunner

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)
    
    # model - parameters, and necessary math algorithms. 
    # model_client/model_server - optimizer, data, training procedure.
    # client_manager/server_maneger - messages to exchange with.
    
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
