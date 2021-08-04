%matplotlib qt5
import matplotlib.pyplot as plt
import numpy as np
import DataManager as dm


nums = ["0_2", "0_3", "0_4", "0_5", "0_6", "0_7", "0_8", "0_9", "1_0"]
p = []
for num in nums:
    p.append(dm.load_data("data/gated_accuracy_varying_thresholds/t_"+num))

fig, ax = plt.subplots()
for i in range(0 , len(nums)):
    #p[i]['simulation'][0]['results']['accuracy'].append(98.5)
    #p[i]['simulation'][0]['results']['energy'].append(p[i]['simulation'][0]['results']['energy'][-1])
    plt.plot(p[i]['simulation'][0]['results']['accuracy'], p[i]['simulation'][0]['results']['error'])
    #ax.annotate(xy=(p[i]['simulation'][0]['results']['accuracy'][-1],p[i]['simulation'][0]['results']['energy'][-1]), xytext=(5,0), textcoords='offset points', text=str(np.around(p[i]['simulation'][0]['results']['accuracy'][-2], 2))+"%", va='center')
    
plt.title("Varying Error Threshold: Accuracy vs Error", fontsize=24)
plt.xlabel("Accuracy", fontsize=18)
plt.ylabel("Error", fontsize=18)
#plt.yscale("log")
plt.legend(["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=16)
plt.show()
