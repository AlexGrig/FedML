import pickle
from collections import defaultdict

class MetricsLogger():
    """
    - Writes down several metrics, some after each minibatch, some after each epoch
    - On several data: minibatch, whole train, whole validation
    """
    def __init__(self, total_samples_no=None, batch_size=None):
        self.N = total_samples_no
        self.bs = batch_size

        self._metrics_attr_prefix = '_metric_'
        self._tracking_metrics = []
        self._attr_time = None
            
    def append(self, key, value, epoch_no=None, mb_num=None, abs_mb_num=None):
        """
        Appends the `value` to corresponding metrics list `key`.
        You should provide either epoch and mb_num, or abs_mb_num
        Args:
            key (str): label of list
            value (str): value to be written 
            epoch_no (int): epoch no
            mb_num (int): batch number within epoch
            abs_mb_num (int): abroslute batch number
        
        Returns:
            None
        """
        
        attr_name = self._key_to_attr_name(key)
        attr = getattr(self, attr_name, None) 
        if attr is None:
            attr = defaultdict(list)
            setattr(self, attr_name, attr)
            self._tracking_metrics.append(key) 
            
        attr[key].append(value)

        # Name of the time attribute
        if epoch_no is not None:
            if mb_num is None:
                attr_time = 'Epoch'; attr_time_val = epoch_no
            else:
                attr_time = 'Epoch_mb'; attr_time_val = (epoch_no, mb_num)
        else:
            attr_time = 'abs_mb'; attr_time_val = abs_mb_num
        
        attr[attr_time].append(attr_time_val)
        
        #if (self._attr_time is None):
        #    self._attr_time = attr_time
        #    setattr(self,self._attr_time, [attr_time_val,])
            
        #if (self._attr_time != attr_time):
        #    class_name = str(MetricsLogger.__class__).split('.')[-1]
        #    raise ValueError( f'{class_name}: time argument mismatch.')
        
        #attr = getattr(self, self._attr_time)
        #if attr[-1] != attr_time_val:
        #    attr.append(attr_time_val)
    
    def _key_to_attr_name(self, key):
        attr_name = self._metrics_attr_prefix + key
        return attr_name
    
    def _verify_all_timestamps_equal(self,):
        
        previous_time_values = None
        for metric_name in self._tracking_metrics:
            attr_name = self._key_to_attr_name(metric_name)
            metric_dict = getattr(self, attr_name)
            time_key = [kk for kk in list(metric_dict.keys()) if kk!=metric_name][0]

            time_values = metric_dict[time_key]
            if previous_time_values is not None:
                if previous_time_values != time_values:
                    return False, previous_time_values, time_key
            else:
                previous_time_values = time_values
        
        return True, previous_time_values, time_key
            
            
    def get_last_time(self,):
        
        time_scales_equal, time_values, time_key = self._verify_all_timestamps_equal()
        if time_scales_equal:
            return time_values[-1], time_key
        else:
            raise ValueError(f'{self.__class__},  get_last_time: Metrics have different timestamps')
    
    def get_last_metrics_dict(self,):
        
        time_scales_equal, time_values, time_key = self._verify_all_timestamps_equal()
        if time_scales_equal:
            return_dict = {}
            for metric in self._tracking_metrics:
                attr_name = self._key_to_attr_name(metric)
                metric_dict = getattr(self, attr_name)
                value_list = metric_dict[metric]
                
                return_dict[metric] = value_list[-1]
            return return_dict
            
        else:
            raise ValueError(f'{self.__class__},  get_last_metrics_dict: Metrics have different timestamps')
    
    def save(self, file_name):
        metrics_dict={}
        for aa in dir(self):
            if aa.startswith(self._metrics_attr_prefix):
                metrics_dict[ aa[(len(aa)-len(self._metrics_attr_prefix) + 1):] ] = getattr(self, aa)

        metrics_dict['total_samples_no'] = self.N
        metrics_dict['batch_size'] = self.bs

        with open(file_name, 'wb') as ff:
            pickle.dump(metrics_dict, ff)