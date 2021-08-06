import multiprocessing as mp
import Network, Gated, Activation
import numpy as np
import time
import DataManager as dm

def main(f, parameters, independent_parameters_idx, name, file_name, max_cpus=mp.cpu_count()):
    if max_cpus > mp.cpu_count():
        raise Exception("Too many CPUs! Leave empty for auto-max.")
    data = [] 
    queue = mp.Queue()
    if len(independent_parameters_idx) == 1:
        print(len(parameters[independent_parameters_idx[0]]))
        if len(parameters[independent_parameters_idx[0]]) > max_cpus:
            print("Over!")
            if np.mod(len(parameters[independent_parameters_idx[0]]), max_cpus) != 0:
                print("Not Nicely Divisible!")
                for m in reversed(range(0, max_cpus)):
                    if np.mod(len(parameters[independent_parameters_idx[0]]), m) == 0:
                        print("M: ", m)
                        chunked_parameters = chunks(parameters[independent_parameters_idx[0]], m)
                        break
            else:
                print("Nicely Divisible!")
                chunked_parameters = chunks(parameters[independent_parameters_idx[0]], max_cpus)
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
                for p in processes:
                    p.join()
                data = data + results
            dm.save_data(file_name, data)           
        else:
            print("Not Over!")
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
            for p in processes:
                p.join()
            data.append(results)
            dm.save_data(file_name, data)
    else:
        print(len(parameters[independent_parameters_idx[1]]))
        if len(parameters[independent_parameters_idx[0]]) > max_cpus:
            print("Over!")
            if np.mod(len(parameters[independent_parameters_idx[0]]), max_cpus) != 0:
                print("Not Nicely Divisible!")
                for m in reversed(range(0, max_cpus)):
                    if np.mod(len(parameters[independent_parameters_idx[0]]), m) == 0:
                        print("M: ", m)
                        chunked_parameters = chunks(parameters[independent_parameters_idx[0]], m)
                        break
            else:
                print("Nicely Divisible!")
                chunked_parameters = chunks(parameters[independent_parameters_idx[0]], max_cpus)
            for i in range(0, len(parameters[independent_parameters_idx[1]])):
                simulations = []
                temp = []
                for j in range(0, len(chunked_parameters)):
                    simulations = []
                    for k in range(0, len(chunked_parameters[j])):
                        simulation = []
                        for n in range(0, len(parameters)):
                            if n == independent_parameters_idx[0]:
                                simulation.append(parameters[n][k])
                            elif n == independent_parameters_idx[1]:
                                simulation.append(parameters[n][i])
                            else:
                                simulation.append(parameters[n])
                        simulations.append(simulation)
                    processes = [mp.Process(target=f, args=(queue, sim, name)) for sim in simulations]
                    for p in processes: 
                        p.daemon = True
                        p.start()
                    results = [queue.get() for p in processes]
                    for p in processes:
                        p.join()
                    temp = temp + results
                data.append(temp)
            dm.save_data(file_name, data)                
        else:
            print("Not Over!")
            for i in range(0, len(parameters[independent_parameters_idx[1]])):
                simulations = []
                for v in range(0, len(parameters[independent_parameters_idx[0]])):
                    simulation = []
                    for k in range(0, len(parameters)):
                        if k == independent_parameters_idx[0]:
                            simulation.append(parameters[k][v])
                        elif k == independent_parameters_idx[1]:
                            simulation.append(parameters[k][i])
                        else:
                            simulation.append(parameters[k])
                    simulations.append(simulation)
                processes = [mp.Process(target=f, args=(queue, sim, name)) for sim in simulations]
                for p in processes:
                    p.daemon = True
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



    
