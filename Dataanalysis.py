# Data Visualization 
import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Clean Text based on https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27
class CleanText(BaseEstimator, TransformerMixin):
       
    def remove_punctuation(self, input_text):
        return re.sub('[^\w\s]+','', input_text)
        
        
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X

      
     
        

def readData_addSentiment():
    
    # Read data
    df=pd.read_excel('../sentences_with_sentiment.xlsx')
    
     

    # Add an additional column Sentiment (Positive== 0, Negative == 1, Neutral == 2)
    sentiment=np.zeros((df.shape[0],1), dtype='i')
    for i in range(0,len(sentiment)):
        if df['Neutral'][i]==1:
            sentiment[i]=2
        if df['Negative'][i]==1:
                sentiment[i]=1
       
        
    df['Sentiment']=sentiment

    # Drop Duplicates if Sentence and Sentiment are equal
    df=df.drop_duplicates(subset=['Sentence', 'Sentiment'])
        
    # Clean test
    ct = CleanText()
    df['clean_text'] = ct.fit_transform(df['Sentence'])

    return df

def frequenceVisualization(statements):
    from nltk.probability import FreqDist 
    

    fd_pos = FreqDist()
    for statement in statements:
        words = statement.split(" ")
        for word in words:
            if word.isalpha():
                fd_pos[word.lower()] += 1
    vocab = fd_pos.keys()
    print(len(vocab))
    print(fd_pos.most_common(20))
    plt.figure(figsize=(16,5))
    fd_pos.plot(20)

def countplot():
    
    cntplt = sns.countplot(x='Sentiment', data=df, color='blue')
    cntplt.set_xticklabels(["Positive","Negative","Neutral"])
    cntplt.get_figure().savefig("Countplot.png")
    plt.clf()
    
if __name__ == "__main__": 
     
    df=readData_addSentiment()
    
    countplot()
    

    # Split Data
    positive_sent_stmts = df.loc[df['Sentiment'] == 0]['clean_text']
    #print(len(positive_sent_stmts))
    negative_sent_stmts = df.loc[df['Sentiment'] == 1]['clean_text']
    #print(len(negative_sent_stmts))
    neutral_sent_stmts = df.loc[df['Sentiment'] == 2]['clean_text']
    #print(len(neutral_sent_stmts))

    # Word frequency analysis
    #frequenceVisualization(df['clean_text'])
    #frequenceVisualization(positive_sent_stmts)
    #frequenceVisualization(negative_sent_stmts)
    #frequenceVisualization(neutral_sent_stmts)



