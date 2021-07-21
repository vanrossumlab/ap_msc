import json

def save_data(file_name, data_dict):
    with open(file_name + ".json", "w") as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)
        
def load_data(file_name):
    with open(file_name + ".json", "r") as f:
        data = json.load(f)
    return data
        
def prepare_simulation_data(name, layers, activation_function, learning_rate, p_connect, bias, n_synapses, 
                            error, accuracy, energy, min_energy, samples_seen, comment=""):
    data_dict = {
        "name" : name,
        "network_parameters" : {
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

def prepare_network_data(name, layers, initial_weights, weights, weight_mask, 
                         biases, activation_function, learning_rate, p_connect, 
                         bias, energy, comment=""):   
        data_dict = {
            "name" : name,
            "network" : {
                "layers" : layers,
                "initial_weights" : [w.tolist() for w in initial_weights], # pythonic has its perks
                "weights" : [w.tolist() for w in weights],
                "weight_mask" : [w.tolist() for w in weight_mask],
                "biases" : [b.tolist() for b in biases],
                "activation_function" : activation_function,
                "learning_rate" : learning_rate,
                "p_connect" : p_connect,
                "bias" : bias,
                "energy" : energy
                },
            "comment" : comment
            }
        return data_dict
    
def prepare_all(name, simulation_data, network_data, comment=""):
    data_dict = {
        "name" : name,
        "simulation" : simulation_data,
        "network" : network_data,
        "comment" : comment
        }
    return data_dict

def prepare_experiment_data(name, experiment_data, network_data=[], comment=""):
    data_dict = {
        "name" : name,
        "simulation" : experiment_data, 
        "comment" : comment
        }
    return data_dict
    
    

    

    
    
    