import logging
import torch.nn as nn
from fedml.model.cv.cnn import CNN_DropOut
from fedml.model.cv.darts import genotypes
from fedml.model.cv.darts.model import NetworkCIFAR
from fedml.model.cv.darts.model_search import Network
from fedml.model.cv.efficientnet import EfficientNet
from fedml.model.cv.mnist_gan import Generator, Discriminator
from fedml.model.cv.mobilenet import mobilenet
from fedml.model.cv.mobilenet_v3 import MobileNetV3
from fedml.model.cv.resnet import resnet56
from fedml.model.cv.resnet56 import resnet_client, resnet_server
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.linear.lr import LogisticRegression
from fedml.model.linear.lr_cifar10 import LogisticRegression_Cifar10
from fedml.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow, RNN_FedShakespeare


def create(args, output_dim):
    global model
    model_name = args.model
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "lr" and args.dataset == "cifar10":
        logging.info("LogisticRegression + CIFAR10")
        model = LogisticRegression_Cifar10(32 * 32 * 3, output_dim)
    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN + MNIST")
        model = CNN_DropOut(False)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_FedShakespeare()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        if args.federated_optimizer == "FedGKT":
            client_model = resnet_client.resnet8_56(c=output_dim)
            server_model = resnet_server.resnet56_server(c=output_dim)
            model = (client_model, server_model)
        else:
            model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMpALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE")
    elif model_name == "efficientnet":
        model = EfficientNet()
    elif model_name == "darts" and args.dataset == "cifar10":
        if args.stage == "search":
            criterion = nn.CrossEntropyLoss()
            model = Network(args.init_channels, output_dim, args.layers, criterion)
        elif args.stage == "train":
            genotype = genotypes.FedNAS_V1
            model = NetworkCIFAR(args.init_channels, output_dim, args.layers, args.auxiliary, genotype)
    elif model_name == "GAN" and args.dataset == "mnist":
        gen = Generator()
        disc = Discriminator()
        model = (gen, disc)
    elif model_name == "lenet" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        from .mobile.mnn_lenet import create_mnn_lenet5_model
        
        create_mnn_lenet5_model(args.global_model_file_path)
        model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    elif model_name == "resnet20" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        from .mobile.mnn_resnet import create_mnn_resnet20_model

        create_mnn_resnet20_model(args.global_model_file_path)
        model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    else:
        raise Exception("no such model definition, please check the argument spelling or customize your own model")
    return model

def create2(model_params):
    """
    Args:
        role (_type_): _description_
        model_params (_type_): _description_
    """
    
    def create_model(model_name, input_dim, output_dim, params_dict, kw_dict=None):
        
        if model_name == 'linreg_batch':
            from fedml.model.linear.linreg_batch import LinearRegression_batch
            model = LinearRegression_batch(input_dim, output_dim, **params_dict)
        elif model_name == 'sum':
            from fedml.model.linear.sum import Sum
            model = Sum(**kw_dict)
        else:
            raise NotImplemented('Model {model_name} is not Implemented')

        return model
    
    role = model_params.role
    input_dim = model_params.X_dim
    client_embedding_dim = model_params.client_embedding_dim
    output_dim = model_params.Y_dim
    with_labels = model_params.with_labels
    
    client_model_name = model_params.client_model
    server_model_name = model_params.server_model
    
    if role == 'server':
        client_model = create_model(client_model_name, input_dim + 1, client_embedding_dim, model_params) # server also has its client model
        # its input dim is +1 since the column of 1s is added to the data to model a bias.
        
        server_model = create_model(server_model_name, client_embedding_dim, output_dim, model_params, {'client_model': client_model})
        if with_labels:
            server_model.client_model = client_model
    elif role == 'client':
        client_model = create_model(client_model_name, input_dim, client_embedding_dim, model_params)
    else:
        raise ValueError(f'Role {role} is undefined') 
    return server_model if role == 'server' else client_model