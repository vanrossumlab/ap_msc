import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from components import DataManager as dm

def mov_mean(x, N):
    r = np.convolve(x, np.ones((N,))/N)[(N-1):]
    r[-1] = x[-1]
    for i in range(0, int(np.ceil(N))):
        r[-i] = x[-i]
    return r

def RGB(R, G, B):
    return (R/255, G/255, B/255)

def unitRGB(units, lineQ):
    if units == 100:
        if lineQ:
            c = RGB(255, 85, 0)
        else:
            c = RGB(255, 158, 110)
    elif units == 50:
        if lineQ:
            c = RGB(255, 149, 0)
        else:
            c = RGB(255, 195, 110)
    elif units == 150:
        if lineQ:
            c = RGB(255, 21, 0)
        else:
            c = RGB(255, 122, 110)
    elif units == 300:
        if lineQ:
            c = RGB(65, 97, 255)
        else:
            c = RGB(159, 183, 255)
    elif units == 250:
        if lineQ:
            c = RGB(144, 65, 255)
        else:
            c = RGB(196, 159, 255)                  
    else:
        if lineQ:
            c = RGB(65, 255, 223)
        else:
            c = RGB(159, 255, 231)
    return c

def swapUnits(u):
    if u == 300:
        u = 200
    elif u == 250:
        u = 150
    elif u == 200:
        u = 100
    elif u == 150:
        u = 50
    elif u == 100:
        u = 300
    elif u == 50:
        u = 250
    return u

all_n_units = [50, 100, 150, 200, 250, 300]
#figure_path = 'data/videos/p_connect/gated/'+ str(n_units) +'/'
filenames = []

accs = np.linspace(92.7, 97.5, 500) #np.flip(np.geomspace(97.5, 92.7, 500))
offset = 0
unit_offsets = 0

filenames = []
for k in range(0, len(accs)):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 8)
    plt.xlim([0.0, 240000])
    #plt.ylim([0, 170000])
    marker_size = 25
    line_size = 5
    window_size = 2
    unit_offset = 0
    
    for n_units in all_n_units:
        print(k)
        figure_path = 'data/videos/p_connect/ungated/'+ str(n_units) +'/'
        filename = figure_path + 'frame_'+str(k)+'.png'
        
        base_path = 'data/'
        file_path = 'ungated/'+ str(n_units) +'_units_various_p'
        data = dm.load_data(base_path+file_path)
        if not len(data) > 1:
            data = data[0]
        
        acc = accs[k]
        energies = []
        p_connects = []
        subenergies = []
        
        
        for i in range(0, len(data)):
            p_connects.append(sum(data[i]['network_parameters']['n_synapses']))
            for j in range(0, len(data[i]['results']['energy'])):
                if data[i]['results']['accuracy'][j] > acc:
                    energies.append(data[i]['results']['energy'][j]/sum(data[i]['network_parameters']['n_synapses']))
                    subenergies.append(data[i]['results']['energy'][-1])
                    break
                else:
                    if j == len(data[i]['results']['energy'])-1:
                        energies.append(0)#data[i]['results']['energy'][-1])
                        subenergies.append(data[i]['results']['energy'][-1]/sum(data[i]['network_parameters']['n_synapses']))
                        
        
        order = np.argsort(p_connects)
        energies = np.array(energies)[order]
        subenergies = np.array(subenergies)[order]
        p_connects = np.array(p_connects)[order]
         
        idxs = np.where(energies == 0)[0]
        energies1 = np.delete(energies, idxs)
        p_connects1 = np.delete(p_connects, idxs)
        
        idx = np.argmax(energies > 0)
        idxs = np.arange(0, idx)
        energies2 = energies[idxs]
        for x in range(0, len(idxs)):
            energies2[x] += subenergies[x]
        energies2 = np.append(energies2, energies[idxs[-1]+1])
        p_connects2 = p_connects[idxs]
        p_connects2 = np.append(p_connects2, p_connects[idxs[-1]+1])
        
        energies = np.where(energies == 0, np.max(energies), energies)
        
        p1 = p_connects[np.where(energies == np.min(energies))]
        e1 = energies[np.where(energies == np.min(energies))]
        e_max = np.max(energies) + unit_offset
        energy_max = np.max(energies)
           
        plt.plot(p_connects1, energies1, linestyle='None', marker='.', markersize=marker_size, color=unitRGB(n_units, False), label='_nolegend_')
        h = plt.plot(p_connects1, mov_mean(energies1, window_size), linewidth=line_size, color=unitRGB(n_units, True))
        plt.plot(p_connects2, energies2, linestyle='None', marker='.', markersize=marker_size, color=unitRGB(n_units, False), label='_nolegend_')
        plt.plot(p_connects2, energies2, linewidth=line_size, color=unitRGB(n_units, False), linestyle='dotted', label='_nolegend_')
        #plt.vlines(p1, e1, e_max, linewidth=line_size, linestyle='dashed', color=unitRGB(n_units, False))
        #ax.annotate(xy=(p_connects1[np.where(energies1 == np.min(energies1))], e_max), xytext=(-75,10), textcoords='offset points', text=str(np.around((1-np.min(energies1)/energies1[-1])*100, 1))+"% Saving" , va='center', fontsize=26, color=unitRGB(n_units, True))
        unit_offset += 2000
    
    #plt.show()
    # base_path = 'data/'
    # file_path = 'ungated/'+ str(n_units) +'_units_various_p'
    # data = dm.load_data(base_path+file_path)
    # if not len(data) > 1:
    #     data = data[0]
    # energies = []
    # p_connects = []
    # subenergies = []
    
    
    # for i in range(0, len(data)):
    #     p_connects.append(data[i]['network_parameters']['p_connect'][0])
    #     for j in range(0, len(data[i]['results']['energy'])):
    #         if data[i]['results']['accuracy'][j] > acc:
    #             energies.append(data[i]['results']['energy'][j])
    #             subenergies.append(data[i]['results']['energy'][-1])
    #             break
    #         else:
    #             if j == len(data[i]['results']['energy'])-1:
    #                 energies.append(0)#data[i]['results']['energy'][-1])
    #                 subenergies.append(data[i]['results']['energy'][-1])
                    
    
    # order = np.argsort(p_connects)
    # energies = np.array(energies)[order]
    # subenergies = np.array(subenergies)[order]
    # p_connects = np.array(p_connects)[order]
     
    # idxs = np.where(energies == 0)[0]
    # energies1 = np.delete(energies, idxs)
    # p_connects1 = np.delete(p_connects, idxs)
    
    # idx = np.argmax(energies > 0)
    # idxs = np.arange(0, idx)
    # energies2 = energies[idxs]
    # for x in range(0, len(idxs)):
    #     energies2[x] += subenergies[x]
    # energies2 = np.append(energies2, energies[idxs[-1]+1])
    # p_connects2 = p_connects[idxs]
    # p_connects2 = np.append(p_connects2, p_connects[idxs[-1]+1])
    
    # energies = np.where(energies == 0, np.max(energies), energies)
    
    # p1 = p_connects[np.where(energies == np.min(energies))]
    # e1 = energies[np.where(energies == np.min(energies))]
    # e_max = np.max(energies)
    # energy_max = np.max(energies)
       
    # plt.plot(p_connects1, energies1, linestyle='None', marker='.', markersize=marker_size, color=unitRGB(n_units, False), label='_nolegend_')
    # h = plt.plot(p_connects1, mov_mean(energies1, window_size), linewidth=line_size, color=unitRGB(n_units, True))
    # plt.plot(p_connects2, energies2, linestyle='None', marker='.', markersize=marker_size, color=unitRGB(n_units, False), label='_nolegend_')
    # plt.plot(p_connects2, energies2, linewidth=line_size, color=unitRGB(n_units, False), linestyle='dotted', label='_nolegend_')
    # plt.vlines(p1, e1, e_max, linewidth=line_size, linestyle='dashed', color=unitRGB(n_units, False))
    # ax.annotate(xy=(p_connects1[np.where(energies1 == np.min(energies1))], e_max), xytext=(-75,10), textcoords='offset points', text=str(np.around((1-np.min(energies1)/energies1[-1])*100, 1))+"% Saving" , va='center', fontsize=26, color=unitRGB(n_units, True))
    
    plt.legend(['50', '100', '150', '200', '250', '300'], fontsize=22, loc='lower right')
    plt.title(' No. Synapses vs. Energy @ ' + str(np.around(acc, 1)) + "% Accuracy", fontsize=36)
    plt.xlabel('No. Synapses', fontsize=28)
    plt.xticks(fontsize=24)
    plt.ylabel('Energy /A.U.', fontsize=28)
    plt.yticks(fontsize=24)
    #plt.xscale('log')
    
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

with imageio.get_writer('test.mp4', mode='I', fps=24) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)



 