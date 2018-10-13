import pandas as pa
import numpy as np
import matplotlib.pyplot as plt  

datset=pa.read_csv("Market_Basket_Optimisation.csv",header=None)
#%%
transactions=[]
for i in range(0,7501):
    transactions.append([str(datset.values[i,j]) for j in range(0,20)])
    
#%%
x=[]
for i in range(0,7501):
    y=[]
    for j in range(0,20):
        y.append(str(datset.values[i,j]))
    x.append(y)
    
    #%%
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#%%
results=list(rules)