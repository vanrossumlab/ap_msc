import multiprocessing as mp
import Network, Gated, Activation
import numpy as np
import time
import DataManager as dm

def run(f, parameters, independent_parameters_idx, name, file_name):
    if len(parameters[independent_parameters_idx[0]]) > mp.cpu_count():
        raise Exception("Varying to many values! (need to think of a way to intelligently distribute items)")  
    data = [] 
    if len(independent_parameters_idx) == 1:
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
            p.start()
        for p in processes:
            p.join()
        results = [queue.get() for p in processes]
        data.append(results)
        dm.save_data(file_name, data)
    else:
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
            for p in processes:
                p.join()
            results = [queue.get() for p in processes]
        data.append(results)
        dm.save_data(file_name, data)

    
