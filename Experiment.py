import multiprocessing as mp
import Network, Gated, Activation
import numpy as np
import time
import DataManager as dm

def main(f, parameters, independent_parameters_idx, name, file_name, max_cpus=mp.cpu_count()):
    data = [] 
    if len(independent_parameters_idx) == 1:
        if len(parameters[independent_parameters_idx[0]]) > max_cpus:
            chunked_parameters = chunks(parameters[independent_parameters_idx[0]], max_cpus)
            queue = mp.Queue()
            for i in range(0, len(chunked_parameters)):
                simulations = []
                for j in range(0, len(chunked_parameters[i])):
                    simulation = []
                    for k in range(0, len(parameters)):
                        if k != independent_parameters_idx[0]:
                            simulation.append(parameters[k])
                        else:
                            simulation.append(chunked_parameters[i][j])
                    simulations.append(simulation)
                processes = [mp.Process(target=f, args=(queue, sim, name)) for sim in simulations]
                for p in processes: 
                    p.daemon = True
                    p.start()
                results = [queue.get() for p in processes]
                data = data + results
                for p in processes:
                    p.join()
            dm.save_data(file_name, data)           
        else:
            queue = mp.Queue()
            simulations = []
            for v in range(0, len(parameters[independent_parameters_idx[0]])):
                simulation = []
                for p in range(0, len(parameters)):
                    if p != independent_parameters_idx[0]:
                        simulation.append(parameters[p])
                    else:
                        simulation.append(parameters[p][v])
                simulations.append(simulation)
            processes = [mp.Process(target=f, args=(queue, simulation, name)) for simulation in simulations]
            for p in processes: 
                p.daemon = True
                p.start()
            results = [queue.get() for p in processes]
            data.append(results)
            dm.save_data(file_name, data)
            for p in processes:
                p.join()
    else:
        raise Exception("Bad stuff ahead!")
        for p in parameters[independent_parameters_idx[1]]:
            simulations = []
            for v in parameters[independent_parameters_idx[0]]:
                simulation = []
                for p in range(0, len(parameters)):
                    if p != independent_parameters_idx[0]:
                        simulation.append(parameters[p])
                    else:
                        simulation.append(parameters[p][v])
                simulations.append(simulation)
            processes = [mp.Process(target=f, args=(queue, simulation, name)) for simulation in simulations]
            for p in processes:  
                p.start()
            results = [queue.get() for p in processes]
            for p in processes:
                p.join()
        data.append(results)
        dm.save_data(file_name, data)
        
def chunks(l, n):
    chunked_l = []
    for i in range(0, len(l), n):
        if i == len(l):
            chunked_l.append(l[i:-1])
        else:
            chunked_l.append(l[i:i+n])
    return chunked_l



    
