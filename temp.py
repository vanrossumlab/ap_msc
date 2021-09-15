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
            c = RGB(255, 21, 255)
        else:
            c = RGB(255, 160, 240)
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
    elif units == 200:
        if lineQ:
            c = RGB(46, 203, 255)
        else:
            c = RGB(157, 234, 255)
    else:
        if lineQ:
            c = RGB(31, 31, 31)
        else:
            c = RGB(158, 158, 158)   
    return c

def swapUnits(u):
    if u == 300:
        u = 200
    elif u == 250:
        u = 150
    elif u == 200:
        u == 100
    elif u == 150:
        u = 50
    elif u == 100:
        u = 300
    elif u == 50:
        u = 250
    return u

n_h_units = [50, 100, 150, 200, 250, 300]


figure_path = 'data/figures/gated/'

fig, ax = plt.subplots()
fig.set_size_inches(16, 8)
marker_size = 25
line_size = 5
window_size = 2

energies = []
p_connects = []
subenergies = []
avg = []

for n_units in n_h_units:
    base_path = 'data/'
    file_path = 'ungated/' + str(n_units) + '_units_various_p'
    data = dm.load_data(base_path+file_path)
    if not len(data) > 1:
        data = data[0]
    for i in range(0, len(data)):
        if data[i]['network_parameters']['p_connect'][0] == 1.0:
            full_a = data[i]['results']['accuracy']
            full_e = data[i]['results']['energy']
    
    base_path = 'data/'
    file_path = 'gated/' + str(n_units) + '_units_gated_various_p'
    data = dm.load_data(base_path+file_path)
    if not len(data) > 1:
        data = data[0]
    for i in range(0, len(data)):
        if data[i]['network_parameters']['p_connect'][0] == 1.0: 
            a = data[i]['results']['accuracy']
            e = data[i]['results']['energy']
            l = str(data[i]['network_parameters']['p_connect'][0])
            break
    accs = np.linspace(a[0], a[-1], 100)
    e_save_full = []
    e_save_e = []
    e_save = []
    js = []
    for j in range(0, len(accs)):
        js.append(j)
        for k in range(0, len(full_a)):
            if full_a[k] >= accs[j]:
                print("Ungated: ", np.around(full_a[k], 2))
                e_save_full.append(full_e[k])
                break
    for j in range(0, len(accs)):
        js.append(j)
        for k in range(0, len(a)):
            if a[k] >= accs[j]:
                print("Gated: ", np.around(a[k], 2))
                e_save_e.append(e[k])
                break
    for p in range(0, len(e_save_e)):
        e_save.append((1-e_save_e[p]/e_save_full[p])*100)
    avg.append(sum(e_save)/len(e_save))
     
bars = plt.bar(n_h_units, avg, width=40, color=[unitRGB(50, True), unitRGB(100, True), unitRGB(150, True), unitRGB(200, True), unitRGB(250, True), unitRGB(300, True)])


# Add text annotations to the top of the bars.
# Note, you'll have to adjust this slightly (the 0.3)
# with different data.
for bar in bars:
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.5,
      str(round(bar.get_height(), 1))+"%",
      horizontalalignment='center',
      color=bar.get_facecolor(),
      fontsize=26
  )
 

plt.title('Gated: Energy Saving vs. Units', fontsize=36)
plt.xlabel('No. Units', fontsize=28)
plt.xticks(fontsize=24)
plt.ylabel('Energy Saving /%', fontsize=28)
plt.yticks(fontsize=24)
plt.ylim([0, 60])

filename = 'energy_saving_gated.png'
plt.savefig(figure_path+filename)
filename = 'energy_saving_gated.svg'
plt.savefig(figure_path+filename)
plt.show()
plt.close()
