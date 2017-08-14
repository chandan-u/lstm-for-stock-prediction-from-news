# lstm-for-stock-prediction-from-news
Using lstms to predict the Dow Jones Industrial Average(DJIA Stock Index) from news. 
We predict the decrease or growth of stock using the daily news. 


## Dataset:

Top 25 news headlines over the past 8 years.

Its a binary classification problem: with the target being 1,0; 

1: growth of stock  (DJIA index)
0: no grow in stock (DJIA index)




## Features used:

We used gensim's word2vec to generate the vectors that represent the words.    
Each document is an average of the words that constitute that document.

1. word2vec vectors of size 100: Documents are represented as average of all the word vectors
2. word2vec vectors of size 300: Documents are represented as average of all the word vectors
3. word2vec vectors of size 100: Documents are represented as average of all the ( word vectors * tf-idf weights)
4. word2vec vectors of size 300: Documents are represented as average of all the ( word vectors * tf-idf weights)


## Time-series data:

Since the nature of data is longitudnal, LSTM's best suit for it. We implemented a look back function that inputs data in a historical fashion:

```
  def create_dataset(dataset_X, dataset_y, look_back=1):
      ...
```

If lookback = 2 , the previous two days news is also given as input to the lstm model.   
if lookback = 10, the previous two days news is also given as input to the lstm model.    



