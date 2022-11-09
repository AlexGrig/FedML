from typing_extensions import assert_never
from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message
import fedml.metrics.metrics as metrics
import fedml.metrics.metrics_logger as metrics_logger
import fedml.metrics.metrics_handlers as metrics_handlers
import fedml.metrics.metrics_tracker as metrics_tracker
from functools import partial
import torch
import logging

class LinregBatchServerManager(FedMLCommManager):
    def __init__(self, connection_parans, train_params, trainer, backend="MPI"):
        super().__init__({}, comm = connection_parans.comm, rank=connection_parans.rank, 
                size = connection_parans.world_size, backend=backend)
        print(train_params)
        self._set_training_params(**train_params)
        
        self.trainer = trainer
        assert connection_parans.rank == 0, f"Server rank must be 0, but is {connection_parans.rank}!"
        self._rank = connection_parans.rank
        self._parties_ids = list( range(0,connection_parans.world_size) )
        self._parties_train_predictions = {}
        self._parties_test_predictions = {}
        self._validation_was_done_on_epoch = None
        self._train_error_computed_on_epoch = None
        self._party_in_operation = None
        
        
        self.logger = logging.getLogger(str(self.__class__).split('.')[-1])
        self.logger.setLevel(logging.WARNING)
        
        self.metrics_logger = logging.getLogger('metrics')
        self.metrics_logger.setLevel(logging.INFO)
        
        accuracy_metric = metrics.ComputeAccuracy(force_to_binary=True)
        rmse_metric = metrics.ComputeRMSE()
        auc_metric = metrics.ComputeAUC()
        
        self.metrics_processor = metrics_tracker.MetricsTracker(self.metrics_logger, (auc_metric, rmse_metric, accuracy_metric) )
        
        
        
    def _set_training_params(self, epochs=None, frequency_of_the_test=None, **kw):
        self.max_epochs = epochs
        self.frequency_of_the_test = frequency_of_the_test
        self.epoch = -1
    
    def run(self):
        self.train_step(self._rank)
        #import pdb; pdb.set_trace()
        super().run() # launch_message_receiving
        
    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_TRAIN_PREDICTIONS, self.handle_client_predictions
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_TEST_PREDICTIONS, self.handle_client_test_predictions
        )
    
    # Trainig responce RHS   
    def handle_client_predictions(self, msg_params):
        #import pdb; pdb.set_trace()
        sender_id = msg_params.get_sender_id()
        if sender_id == self._party_in_operation:
            predictions = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_TRAIN_PREDICTIONS)
                
            self._parties_train_predictions[sender_id] = predictions
            self._party_in_operation = None
            
            self.logger.info(f"Server {self._rank}: finished processing client {sender_id} TRAIN predictions. Type: {type(predictions)}, shape: {predictions.shape}")
            
            # Important: send train_step request for the next party.
            next_party = self._topology_next_party_id(sender_id)
            self.train_step(next_party) # Send message to the next client.
        else:
            self.logger.warning(f'{self.__class__}: waiting predictions from party {self._party_in_operation} \
                            but received from {sender_id}. Doing nothing now')
            
    # Trainig request RHS        
    def send_rhs_to_client(self, receive_rank, rhs):
        message = Message(
            MyMessage.MSG_TYPE_S2C_SEND_RHS, self._rank, receive_rank)
        message.add_params(MyMessage.MSG_ARG_KEY_RHS, rhs)
        self.send_message(message)
        self.logger.info(f"Server {self._rank} -> client {receive_rank}: TRAIN request sent.")
        
    # Test responce
    def handle_client_test_predictions(self, msg_params):
        
        sender_id = msg_params.get_sender_id()
        
        predictions = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_TEST_PREDICTIONS)
        self._parties_test_predictions[sender_id] = predictions

        self.logger.info(f"Server {self._rank}: finished processing client {sender_id} TEST predictions. Type: {type(predictions)}, shape: {predictions.shape}")
        
        if len(self._parties_test_predictions) == len(self._parties_ids):
            self._compute_test_metrics()
    
    # Test request
    def send_val_request_to_client(self, receive_rank):
        message = Message(
            MyMessage.MSG_TYPE_S2C_SEND_VAL_REQUEST, self._rank, receive_rank)
        self.send_message(message)
        self.logger.info(f"Server {self._rank} -> client {receive_rank}: TEST request sent.")
    
    def _topology_next_party_id(self, current_party_id):
        """Returns ID of the next party to be updated.

        Args:
            current_party_id (_type_): id of the current party
        """
        parties_amount = len(self._parties_ids)
        last_index = parties_amount - 1
        current_party_index = self._parties_ids.index(current_party_id)
        
        if current_party_index < last_index: # not last index
            return self._parties_ids[current_party_index + 1]
        else:
            return self._parties_ids[0]
        
    def _compute_party_rhs(self,party_id):
        #import pdb;pdb.set_trace()
        sum_rhs = [value for (key,value) in self._parties_train_predictions.items() if (key!=party_id) ]
        if len(sum_rhs) > 1:
            sum_pred_except_party = torch.atleast_2d( torch.sum(torch.concat(sum_rhs, dim=1), dim=1) ).t()
        elif len(sum_rhs) == 1:
            sum_pred_except_party = sum_rhs[0]
        else:
            sum_pred_except_party = 0
        return -sum_pred_except_party
    
    def _update_party(self, party_id):
        if party_id == self._rank:
            self._update_server_party()
        else:
            self._update_client_party(party_id)
            
    def _update_client_party(self, party_id):
        party_rhs = self._compute_party_rhs(party_id)
        self.send_rhs_to_client(party_id, party_rhs)
        self._party_in_operation = party_id
    
    def _update_server_party(self,):
        #import pdb;pdb.set_trace()
        rhs = self.trainer.get_Y_train() + self._compute_party_rhs(self._rank)
        self._party_in_operation = self._rank
        
        weights = self.trainer.update_model_weights(rhs)
        minus_server_residuals = self.trainer.predict_train(return_residuals=True)
        self._parties_train_predictions[self._rank] = minus_server_residuals
        
        self._party_in_operation = None
    
    def train_step(self, party_id):
        if party_id == self._rank: # beginning of the new epoch
            #import pdb; pdb.set_trace()
            train_error_condition = (self._train_error_computed_on_epoch != self.epoch) and (self.epoch != -1)
            test_error_condition = ((self.epoch % self.frequency_of_the_test) == 0) and (self.epoch != -1) and\
                (self._validation_was_done_on_epoch != self.epoch)
                
            if train_error_condition:
                self.logger.info(f'Train metrics computation on epoch {self.epoch} has started.')
                self._compute_train_metrics()
                if not test_error_condition:
                    self.metrics_processor.log_output() # output train metrics
                self.train_step(self._rank) # return to training (or do validation (test))
                return
            elif test_error_condition: # do validation on previous epoch
                self._parties_test_predictions = {}
                self._parties_test_predictions[self._rank] = self.trainer.predict_test(return_residuals=False)
                self.logger.info(f'Validation on epoch {self.epoch} has started.')
                self._launch_test_request()
                return
            else: # start new training epoch
                #if (self._validation_was_done_on_epoch != self.epoch) and (self.epoch != -1): # print metrics if has not been printed before
                #    self.metrics_processor.log_output()
                self.epoch += 1 
                
            #self._parties_predictions = {}
            if self.epoch > self.max_epochs:
                self.logger.info(f'Maximum number of epochs {self.max_epochs} has been completed')
                self.finish_protocol()
            else:
                self.logger.info(f"Epoch {self.epoch} started")
            self._update_server_party()
            next_party_id = self._topology_next_party_id(self._rank)
            self.train_step(next_party_id)
        else:
            self._update_client_party(party_id)
    
    def _launch_test_request(self, ):
        for party_id in self._parties_ids:
            if party_id == self._rank: # server:
                self._parties_test_predictions[self._rank] = self.trainer.predict_test(return_residuals=False)
            else:
                self.send_val_request_to_client(party_id)
                
    def _compute_test_metrics(self,):
        
        predictions = torch.atleast_2d( torch.sum(torch.concat(list(self._parties_test_predictions.values()), dim=1), dim=1) ).t()
        
        self.metrics_processor.append_metrics(self.epoch, self.trainer.get_Y_test(), predictions, metrics_logger_name='test')
        
        self.metrics_processor.log_output()
        
        self._validation_was_done_on_epoch = self.epoch
        self.train_step(self._rank) # return to training
    
    def _compute_train_metrics(self, ):
        #import pdb; pdb.set_trace()
        #predictions = torch.atleast_2d( torch.sum(torch.concat(list(self._parties_train_predictions.values()), dim=1), dim=1) ).t()
        
        sum_clients = -self._compute_party_rhs(self._rank) # exclude server
        server_predictions = self.trainer.predict_train(return_residuals=False)
        
        predictions = sum_clients + server_predictions
        
        self.metrics_processor.append_metrics(self.epoch, self.trainer.get_Y_train(), predictions, metrics_logger_name='train')
        self._train_error_computed_on_epoch = self.epoch
        
        
    def finish_protocol(self):
        for client in [cc for cc in self._parties_ids if cc!= self._rank]:
            self.send_finish_to_client(client)
        self.finish()
        
    def send_finish_to_client(self, receive_id):
        message = Message(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self._rank, receive_id)
        self.send_message(message)
