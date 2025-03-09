import numpy as np

def read_config(path_config="client.properties"):
    config = {}
    with open(path_config) as fh:
        for line in fh:
            line = line.strip()
            if len(line) != 0 and line[0] != "#":
                parameter, value = line.strip().split('=', 1)
                config[parameter] = value.strip()
    return config

def save_to_np_array(file_to_save, data_to_save):
    np_data = np.array(data_to_save)
    
    with open(file_to_save, 'wb') as file:
        np.save(file, np_data)