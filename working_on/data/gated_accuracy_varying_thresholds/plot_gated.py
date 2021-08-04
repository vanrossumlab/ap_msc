import matplotlib.pyplot as plt
import numpy as np
import DataManager as dm


nums = ["0_2", "0_3", "0_4", "0_5", "0_6", "0_7", "0_8", "0_9", "1_0"]
p = []
for num in nums:
    p.append(dm.load_data("t_"+num))

labels = []
#for lr in range(0 , len(nums)):
