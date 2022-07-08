import numpy as np 
import pandas as pd
import nltk
import re
from  nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



df=pd.read_csv("SMSSpamCollection",sep='\t',names=["label","message"])


ps=PorterStemmer()
sms=[]


#simplifing data
for i in range(0,len(df)):
	n=re.sub('[^a-zA-Z]',' ',df['message'][i])
	n=n.lower()
	n=n.split()
	n=ps.stem(words) for words in n if not words in stopwords.words('english')
	n=' '.join(n)
	sms.append(n)



#bag of words
cv=CountVectorizer(max_features=250)
x=cv.fit_transform(sms).toarray()
y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values



#train test split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_seed=0)

#model

model=MultinomialNB().fit(x_train,y_train)
pred=model.predict(x_test)


from sklearn.metrics import accuracy_score
a=accuracy_score(pred,y_test)
print(a)