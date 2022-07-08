import numpy as np 
import pandas as pd





df=pd.read_csv("Combined_News_DJIA.csv")



traindata=df[df['Date']<'20150101']
testdata=df[df['Date']>'20140101']



data=traindata.iloc[:,2:]


index_range=range(0,25)
data.columns=index_range



data.replace('[^a-zA-Z]',' ',regex=True,inplace=True)

for index in index_range:
	data[index]=data[index].str.lower()

news=[]

for i in range (0,len(data)):
	news.append(''.join(str(x) for x in data.iloc[i,:]))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(2,2))
data=cv.fit_transform(news)

test_data=testdata.iloc[:,2:]


index_range=range(0,25)
test_data.columns=index_range



test_data.replace('[^a-zA-Z]',' ',regex=True,inplace=True)

for index in index_range:
	test_data[index]=test_data[index].str.lower()

new=[]

for i in range (0,len(test_data)):
	new.append(''.join(str(x) for x in test_data.iloc[i,:]))

test_data=cv.transform(new)




from sklearn.neighbors import KNeighborsClassifier

score=[]
k_range=range(1,25)

from sklearn.metrics import accuracy_score


for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(data,traindata['Label'])
    pred=knn.predict(test_data)
    score.append(accuracy_score(pred,testdata['Label']))



import matplotlib.pyplot as plt

plt.plot(k_range,score)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()


