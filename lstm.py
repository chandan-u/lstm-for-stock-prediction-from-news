
from __future__ import print_function


import pandas as pd
import numpy as np
import math


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import re
import string

regex = re.compile('[%s]' % re.escape(string.punctuation))
regexb=re.compile('b[\'\"]')
stop_words = set(stopwords.words('english'))


import tensorflow as tf
    

from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation
from tensorflow.contrib.keras.python.keras.layers.core import Reshape

def loadDataCombinedColumns(path='../datasets/Combined_News_DJIA.csv'):
    """
      Combine all the news headlines(25 columns) into one. 
      All the headlines for any given day is treated as one document
      Split as per dates: 80 percent train and 20 percent test
      replace nan with empty strings
      make date as an index (helps with time series data)
    """
    data = pd.read_csv(path, parse_dates = True, index_col = 0, verbose =True, keep_default_na=False) 
    data_y = data["Label"]
    data_X = data.iloc[:,1:26].apply(lambda headline:cleanString(' '.join(headline)), axis=1)

    
    test_X  = data_X['2015-01-02':'2016-07-01']
    train_X = data_X['2008-08-08':'2014-12-31']
    test_y  = data_y['2015-01-02':'2016-07-01']
    train_y = data_y['2008-08-08':'2014-12-31']
    return (train_X, train_y,  test_X, test_y)


 
def cleanString(sentence):
    """
        get Grams for a  sentence
        Custom Function

    """
    return ' '.join(getGrams(sentence))


def getGrams(sentence):
    """
        get Grams for a  sentence
        Custom Function

    """
    sentence = sentence.lower()
    sentence = regexb.sub('', sentence)
    sentence = regex.sub('', sentence)
    tokens = filter(lambda token: token != '', word_tokenize(sentence))
    #tokens = filter(lambda word: word not in stop_words, tokens)

    
    #return filter(lambda word: word not in stop_words, filter(lambda token: token != '', map(lambda token:regex.sub('', token),map(str.lower, word_tokenize(sentence)))))

    return tokens

def create_dataset(dataset_X, dataset_y, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset_X)-look_back-1):
        a = dataset_X[i:(i+look_back),: ]
        dataX.append(a)
        dataY.append(dataset_y.iloc[i + look_back])
    return np.array(dataX), np.array(dataY)



train_X, train_y, test_X, test_y  = loadDataCombinedColumns()



# pre built features    
w2v_tfIdf_train_X, w2v_tf_Idf_test_X = np.loadtxt("../datasets/w2v_100_tfIdf_train_X.txt"), np.loadtxt("../datasets/w2v_100_tfIdf_test_X.txt")
        

"""
  https://keras.io/getting-started/sequential-model-guide/
  A good source to understand sequential models in Keras including lstms 

"""

look_back = 5
data_dim = 100

model = Sequential()

# if return_sequences = Flase, only last LSTM cell will slpit the output.
model.add(LSTM(32, input_shape =(look_back,data_dim), return_sequences=True ))

model.add(LSTM(2))
#model.add(Dense(10))

model.add(Dropout(0.2))

model.add(Dense(1, activation='softmax'))

# try using different optimizers and different optimizer configs

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])




time_train_X, time_train_y = create_dataset(w2v_tfIdf_train_X, train_y, look_back = look_back)

time_test_X, time_test_y = create_dataset(w2v_tf_Idf_test_X, test_y, look_back = look_back)




model.fit(time_train_X, time_train_y,verbose=2,epochs=10,batch_size=100, validation_data=[time_test_X, time_test_y])

        
