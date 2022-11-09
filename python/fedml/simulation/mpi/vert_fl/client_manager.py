import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class LinregBatchClientManager(FedMLCommManager):
    def __init__(self, connection_parans, train_params, trainer, backend="MPI"):
        super().__init__({}, comm = connection_parans.comm, rank=connection_parans.rank, 
                size = connection_parans.world_size, backend=backend)
        self.trainer = trainer
        self._rank = connection_parans.rank
        self._server_rank = 0
        self.logger = logging.getLogger(str(self.__class__).split('.')[-1])
        self.logger.setLevel(logging.WARNING)
        
    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SEND_RHS, self.handle_receive_rhs)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SEND_VAL_REQUEST, self.handle_test_request)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.handle_message_finish_protocol)

    # 
    def handle_receive_rhs(self, msg_params):
        #import pdb; pdb.set_trace()
        sender_id = msg_params.get_sender_id()
        
        rhs = msg_params.get(MyMessage.MSG_ARG_KEY_RHS)
        self.trainer.update_model_weights(rhs)

        self.send_train_predictions(self._server_rank)
        self.logger.debug(f"Client {self._rank}: received TRAIN request from {sender_id}. RHS type: {type(rhs)}, shape: {rhs.shape}. Responce sent back.")
        
        weights = self.trainer._get_weigths()
        self.logger.info(f'Client {self._rank}: weights: {weights}')
        
    def handle_test_request(self,msg_params):
        
        sender_id = msg_params.get_sender_id()
        predictions = self.trainer.predict_test()
        
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_TEST_PREDICTIONS, self._rank, sender_id)
        
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_TEST_PREDICTIONS, predictions)
        self.send_message(message)
        self.logger.debug(f"Client {self._rank}: received TEST request from {sender_id}. Responce sent back.")
    
    def send_train_predictions(self, receiver_rank):
        predictions = self.trainer.predict_train()
        
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_TRAIN_PREDICTIONS, self._rank, receiver_rank)
        
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_TRAIN_PREDICTIONS, predictions)
        self.send_message(message)
    
    def handle_message_finish_protocol(self, msg_params):
        sender_id = msg_params.get_sender_id()
        self.logger.debug(f"Client {self._rank}: received STOP SIMULATION request from {sender_id}")
        self.finish()