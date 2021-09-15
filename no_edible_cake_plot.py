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

base_path = 'data/'
file_path = 'no_edible_cake'
data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]

fig, ax = plt.subplots()
fig.set_size_inches(16, 8)
plt.plot(data[0]['results']['accuracy'][1], data[0]['results']['energy'][1], color=unitRGB(300, True), linewidth=5)
plt.plot(data[1]['results']['accuracy'][1], data[1]['results']['energy'][1], color=unitRGB(0, True), linewidth=5)
plt.legend(['Ungated: 1.0', 'Gated: 1.0'], fontsize=22, loc='lower right')
plt.title('Ungating @ 96% Accuracy 150 Units: Energy vs. Accuracy' , fontsize=36)
plt.xlabel('Accuracy /%', fontsize=28)
plt.xticks(fontsize=24)
plt.ylabel('Energy /A.U.', fontsize=28)
plt.yscale('log')
plt.yticks(fontsize=24)
plt.xlim([84, 100])
plt.ylim([1000, 250000])

figure_path = 'data/figures/gated/'
filename = 'ungating_at_96.svg'
plt.savefig(figure_path+filename)
filename = 'ungating_at_96.png'
plt.savefig(figure_path+filename)
plt.close()
