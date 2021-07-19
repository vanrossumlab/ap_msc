import json

def save_data(file_name, data_dict):
    with open(file_name + ".json", "w") as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)
        
def load_data(file_name):
    with open(file_name + ".json", "r") as f:
        data = json.load(f)
    return data
        
def prepare_experiment_data(name, layers, activation_function, learning_rate, p_connect, bias, n_synapses, 
                            error, accuracy, energy, min_energy, samples_seen, comment=""):
    data_dict = {
        "name" : name,
        "network parameters" : {
            "layers" : layers,
            "activation_function" : activation_function,
            "learning_rate" : learning_rate,
            "p_connect" : p_connect,
            "bias" : bias,
            "n_synapses" : n_synapses.tolist()
            },
        "results" : {
            "error" : error.tolist(),
            "accuracy": accuracy.tolist(),
            "energy" : energy.tolist(),
            "min_energy" : min_energy.tolist(),
            "samples_seen" : samples_seen.tolist()
            },
        "comment" : comment
        }
    return data_dict

def prepare_network_dict(name, layers, initial_weights, weights, weight_mask, 
                         biases, activation_function, learning_rate, p_connect, comment=""):
        data_dict = {
            "name" : name,
            "network" : {
                "layers" : layers,
                "initial_weights" : initial_weights,
                "weights" : weights,
                "weight_mask" : weight_mask,
                "biases" : biases,
                "activation_function" : activation_function,
                "learning_rate" : learning_rate,
                "p_connect" : p_connect
                },
            "comment" : comment
            }
        return data_dict
    
def prepare_all(name, experiment_data_dict, network_data_dict, comment=""):
    data_dict = {
        "name" : name,
        "experiment_data" : experiment_data_dict,
        "network_data" : network_data_dict,
        "comment" : comment
        }
    return data_dict
    
    

    

    
    
    