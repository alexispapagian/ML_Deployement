import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

#convert words to integer values
def convert_to_int(word):
    word_dict={'one':1, 'two': 2, 'three':3, 'four':4 ,'five':5,
               'six': 6 , 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11,
               'zero':0, 0:0}
    return word_dict[word]

if __name__ == "__main__":

    dataset= pd.read_csv('hiring.csv')

    dataset['experience'].fillna(0,inplace=True)

    dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

    print(dataset)

    X=dataset.iloc[:,:3]
    print(X)

    X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

    y = dataset.iloc[:,-1]
    
    # Splitting Training and test set
    # Since we have small dataset, all will be training 

    regressor = LinearRegression()

    # Fitting model with training data
    regressor.fit(X,y)
    
    # Saving model to disk
    pickle.dump(regressor, open('model.pkl','wb'))

    # Loading model to compare results
    model = pickle.load(open('model.pkl','rb'))
    print(model.predict([[2,0,6]]))
