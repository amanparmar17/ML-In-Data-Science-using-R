import pandas as pa
import numpy as np
import matplotlib.pyplot as plt  

datset=pa.read_csv("Ads_CTR_Optimisation.csv")
#datset contions a matrix of 0s and 1s, which are nothing but the rewards corresponding to each option at each round

#%%implementing the ucb reinforcement method

import math

N=10000         #no of rounds
d=10            #no of options available
                #n is the ierator of the number of round
                #i is the iterator of the number of option
number_of_selections= [0]*d         #vector of lenght =options containing the number of times 
                                    #each of the option is selected
sum_of_rewards=[0]*d                #vector containing the sum of rewards for each of the options available
ads_selected=[]                     #contains the option nmber selected at each round,
                                    #i.e at the end of the algo, it will be a list of 10000 elements
total_reward=0                      #contains the total reward provided at the end of the algo(i.e. 10000 rounds)
   
                                    #as initially each of the option is selected 0 times and also the sum of the rewards for each of the option is 0
                                    #both the vectors are initialised to 0
for n in range(0,N):            #for the first o=10 rounds, at each round o, oth option is selected,providing the algo some initial innfo to start wwith  
    upper_bound_list=[]         #contains the ucb for each option at each round
    for i in range(0,d):
        if number_of_selections[i]>0:           #since we have no info in the beginning, each option has to be selected atleast ones to provide the algo with some info
            average_reward=sum_of_rewards[i]/number_of_selections[i]
            delta_i=math.sqrt((3/2)*(math.log(n+1)/number_of_selections[i]))
            upper_bound=average_reward+delta_i
            upper_bound_list.append(upper_bound)
        else:                                       #if th round is no tselected even ones
            upper_bound=1e400                #upper bound to a very large value so thaat this option gets selected(value = 10^400........ math.pow(10,400) could not work due to range limit)
            upper_bound_list.append(upper_bound)
    x=upper_bound_list.index(max(upper_bound_list))     #find the index of option with the max ucb
    ads_selected.append(x)                              #append the selected option to the list
    number_of_selections[x]=number_of_selections[x]+1   #update the selection count for the selectedd option 
    reward=datset.values[n,x]                                  #select the reward for the selected opt(index x) from the dataset
    sum_of_rewards[x]=sum_of_rewards[x]+reward          #update the reward vector for the sum of reward of each option
    total_reward=total_reward+reward
    
#%%
plt.hist(ads_selected)
plt.show()