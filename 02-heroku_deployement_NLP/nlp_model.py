import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1,inplace=True)
df['label'] = df['class'].map({'ham':0,'spam':1})
X = df['message']
y = df['label']

#extract feature with CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv, open('transform.pkl','wb'))

X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.33, random_state=42)


clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
pickle.dump(clf, open('nlp_model.pkl','wb'))