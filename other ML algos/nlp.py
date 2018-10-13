import pandas as pa
import numpy as np
import matplotlib.pyplot as plt  

datset=pa.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)
#%%
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review= re.sub('[^a-zA-Z]',' ',datset['Review'][i])
    review=review.lower()
    review=review.split()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)
#%% bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=datset.iloc[:,1].values


#%%classification
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20, random_state=0)

#%%
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(xtrain,ytrain)
#%%
ypred=classifier.predict(xtest)
#%%
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)

