from . import metrics_logger
from . import metrics_handlers

class MetricsTracker:
    def __init__(self, logger, metrics_processors, metrics_loggers=('train', 'test')):
        self.logger = logger
        
        self._metrics_processors = metrics_processors
        self._metrics_loggers = metrics_loggers
        
        for ml in metrics_loggers:
            ml_attr_name = self._logger_name_to_attr_name(ml)
            setattr(self,ml_attr_name, metrics_logger.MetricsLogger())
        
    def _logger_name_to_attr_name(self,name): 
        attr_name = '_metrics_logger_' + name
        return attr_name
    
    def append_metrics(self,epoch, true_labels, predictions, metrics_logger_name='train'):
        
        ml_attr_name = self._logger_name_to_attr_name(metrics_logger_name)
        metrics_logger = getattr(self, ml_attr_name)
        
        for metric in self._metrics_processors:
            metric_val  = metric.compute(true_labels,predictions)
            metrics_logger.append(metric.name, metric_val, epoch_no=epoch, mb_num=None, abs_mb_num=None)
            
    def log_output(self,):
        #import pdb; pdb.set_trace()
        train_ml_name  = [ml for ml in self._metrics_loggers if 'train' in ml.lower()][0]
        test_ml_name = [ml for ml in self._metrics_loggers if 'test' in ml.lower()][0]
        
        train_metrics_logger = getattr(self, self._logger_name_to_attr_name(train_ml_name) )
        test_metrics_logger = getattr(self, self._logger_name_to_attr_name(test_ml_name) )
        
        train_last_time, time_attr = train_metrics_logger.get_last_time()
        test_last_time, _ = test_metrics_logger.get_last_time()
        
        if (test_last_time is None) or (test_last_time < train_last_time):
            metrics_dict = self._output_metrics(metrics_logger_name='train')
            output_string = metrics_handlers.logging_handler(f' {time_attr}: {train_last_time}, TRAIN: ', metrics_dict)
            
        elif test_last_time > train_last_time:
            raise ValueError('Metrics logging improper use')
            
        elif test_last_time == train_last_time:
            metrics_dict = self._output_metrics(metrics_logger_name='train')
            metrics_dict_test = self._output_metrics(metrics_logger_name='test')
            
            output_string = metrics_handlers.logging_handler(f' {time_attr}: {test_last_time}, TRAIN: ', metrics_dict)
            
            output_string_test = metrics_handlers.logging_handler('        TEST:', metrics_dict_test )
            output_string += output_string_test
        else:
            raise ValueError('')
        
            #output_string =  f' {time_attr}: {train_last_time}  {metrics_dict},  {metrics_dict_test}'
        
        #output_string =  f' {time_attr}: {test_last_time},   {metrics_dict_test}'
        
        #output_string = metrics_handlers.logging_handler(f' {time_attr}: {test_last_time}, ', metrics_dict_test)
        
        self.logger.info(output_string)
        
    def _output_metrics(self, metrics_logger_name='train'):
        
        ml_attr_name = self._logger_name_to_attr_name(metrics_logger_name)
        metrics_logger = getattr(self, ml_attr_name)
            
        metrics_dict = metrics_logger.get_last_metrics_dict()
        
        return metrics_dict
