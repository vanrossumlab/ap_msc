import numpy as np
import matplotlib.pyplot as plt
from components import DataManager as dm

def normalise(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def diff_eq(x):
    temp = np.zeros(len(x))
    for i in range(0, len(x)-1):
        temp[i] = x[i+1] - x[i]
    return temp
            

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

base_path = 'data/'
file_path = 'training_curve'
data = dm.load_data(base_path+file_path)
marker_size = 25
line_size = 5

s = np.array(data[0][1]['results']['samples_seen'][0])
a = np.array(data[0][1]['results']['accuracy'][0])
tr = np.array(data[0][1]['results']['samples_seen'][1])
tr = diff_eq(tr)
t = tr/1000
t = t*255

marker_size = 500
fig, ax = plt.subplots()
fig.set_size_inches(16, 8)
colours = []
for i in range(0, len(a)):
    plt.scatter(s[i], a[i], marker='.', s=marker_size, color=unitRGB(0, True, t[i]))
    colours.append(unitRGB(0, True, t[i]))
plt.title('Gated 100 Units: Samples Trained On vs. Accuracy ', fontsize=36)
plt.xlabel('Samples (Epochs)', fontsize=28)
plt.xticks(fontsize=24)
plt.ylabel('Energy /a.u.', fontsize=28)
plt.yticks(fontsize=24)
te = ax.xaxis.get_offset_text()
te.set_size(20)
plt.xlim([0, 3e6])
plt.ylim([85, 100])
figure_path = 'data/figures/gated/'
filename = 'gated_100_units_trained_on.svg'
fig.tight_layout()
plt.savefig(figure_path+filename, bbox_inches='tight')
filename = 'gated_100_units_trained_on.png'
plt.savefig(figure_path+filename, bbox_inches='tight')
plt.show()