def logging_handler(prefix_string, metrics_dict):
    log_string = prefix_string
    for key in metrics_dict:
        value = metrics_dict[key]
        text = f' {key}: {value:.8f}'
        log_string += text
        
    return log_string
    
    