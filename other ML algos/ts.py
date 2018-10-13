import pandas as pa
import numpy as np
import matplotlib.pyplot as plt  

datset=pa.read_csv("Ads_CTR_Optimisation.csv")
#datset contions a matrix of 0s and 1s, which are nothing but the rewards corresponding to each option at each round

#%%implementing the ucb reinforcement method

import random

N=10000         #no of rounds
d=10            #no of options available
                #n is the ierator of the number of round
                #i is the iterator of the number of option
number_of_rewards_1=[0]*d      #vector of lenght =options containing the number of times 
                                    #each of the option got reward 1
number_of_rewards_0=[0]*d           #vector of lenght =options containing the number of times each of the option got reward 0
ads_selected=[]                     #contains the option nmber selected at each round,
                                    #i.e at the end of the algo, it will be a list of 10000 elements
total_reward=0                      #contains the total reward provided at the end of the algo(i.e. 10000 rounds)
   
                                    #as initially each of the option is not provided a single reward(0/1)
                                    #both the vectors are initialised to 0
for n in range(0,N):            #for the first o=10 rounds, at each round o, oth option is selected,providing the algo some initial innfo to start wwith  
    random_draw_list=[]         #contains the random draws made for each option at each round
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
                                # rnnadom draw for each of the option is equal to th ebetavariate of the 
                                #number_of_rewards_1[i] and number_of_rewards_0[i]
        random_draw_list.append(random_beta)    #append the random_draaw value of reach option in the list
         
    x=random_draw_list.index(max(random_draw_list))     #find the index of option with the max random draw
    ads_selected.append(x)                              #append the selected option to the list
    
    reward=datset.values[n,x]                                  #select the reward for the selected opt(index x) from the dataset
    if reward==1:                                               #if reward of rht eselected option from the selected option is 1,increment number_of_rewards_1[i]
        number_of_rewards_1[x]=number_of_rewards_1[x]+1
    else:                                                       #if reward of rht eselected option from the selected option is 0,increment number_of_rewards_0[i]
        number_of_rewards_0[x]=number_of_rewards_0[x]+1
    total_reward=total_reward+reward                                #total reward at the end of the algorithm
    
#%%
plt.hist(ads_selected)
plt.show()           