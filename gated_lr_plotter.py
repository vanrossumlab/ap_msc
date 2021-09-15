import numpy as np
import matplotlib.pyplot as plt
from components import DataManager as dm

def mov_mean(x, N):
    r = np.convolve(x, np.ones((N,))/N)[(N-1):]
    r[-1] = x[-1]
    for i in range(0, int(np.ceil(N))):
        r[-i] = x[-i]
    return r

def RGB(R, G, B, A=255):
    return (R/255, G/255, B/255, A/255)

def unitRGB(units, lineQ, opacity=255):
    if units == 100:
        if lineQ:
            c = RGB(255, 21, 255, opacity)
        else:
            c = RGB(255, 160, 240, opacity)
    elif units == 50:
        if lineQ:
            c = RGB(255, 149, 0, opacity)
        else:
            c = RGB(255, 195, 110, opacity)
    elif units == 150:
        if lineQ:
            c = RGB(255, 21, 0, opacity)
        else:
            c = RGB(255, 122, 110, opacity)
    elif units == 300:
        if lineQ:
            c = RGB(65, 97, 255, opacity)
        else:
            c = RGB(159, 183, 255, opacity)
    elif units == 250:
        if lineQ:
            c = RGB(144, 65, 255, opacity)
        else:
            c = RGB(196, 159, 255, opacity)              
    elif units == 200:
        if lineQ:
            c = RGB(46, 203, 255, opacity)
        else:
            c = RGB(157, 234, 255, opacity)
    else:
        if lineQ:
            c = RGB(31, 31, 31, opacity)
        else:
            c = RGB(158, 158, 158, opacity)   
    return c

fp = 'data/various_lr_p'

data = dm.load_data(fp)

labs = []
j = 1
lrs = [0.0005, 0.001, 0.005, 0.01, 0.05]
#for j in range(len(data)):
n = 0
m = 50
fig, ax = plt.subplots()
fig.set_size_inches(16, 8)
for k in range(0, len(data[0])):
    for i in range(len(data[0])):
        if int(data[j][i]['network_parameters']['learning_rate'][0]*10000) == int(lrs[n]*10000):
            plt.plot(data[j][i]['results']['accuracy'][1], data[j][i]['results']['energy'][1], color=unitRGB(300, True, m), linewidth=4)
            labs.append(str(data[j][i]['network_parameters']['learning_rate'][0]))
            n = n + 1
            m = m + 50
            break
     
plt.title('Ungated: Energy vs. Accuracy with 100 Units\n@ Various Learning Rates', fontsize=36)
plt.xlabel('Accuracy /%', fontsize=28)
plt.xticks(fontsize=24)
plt.ylabel('Energy /A.U.', fontsize=28)
plt.yscale('log')
plt.yticks(fontsize=24)
plt.xlim([84, 100])
plt.ylim([2000, 250000])
plt.legend(labs, fontsize=22, loc='lower right')
plt.xlim(88, 100)
plt.ylim(4000, 300000)
plt.yscale('log')
figure_path = 'data/figures/ungated/'
filename = 'ungated_accuracy_vs_energy_various_lr2.svg'
fig.tight_layout()
plt.savefig(figure_path+filename, bbox_inches='tight')
filename = 'ungated_accuracy_vs_energy_various_lr2.png'
plt.savefig(figure_path+filename, bbox_inches='tight')   
plt.show()
