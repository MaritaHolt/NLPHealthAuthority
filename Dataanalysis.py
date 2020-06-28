# Data Visualization 

def readData_addSentiment():
    import pandas as pd 
    import numpy as np
    # Read data
    df=pd.read_excel('../sentences_with_sentiment.xlsx')
    df=df.drop(columns=['ID'])

    # Add an additional column Sentiment (Positive== 0, Negative == 1, Neutral == 2)
    sentiment=np.zeros((df.shape[0],1), dtype='i')
    for i in range(0,len(sentiment)):
        if df['Neutral'][i]==1:
            sentiment[i]=2
        if df['Negative'][i]==1:
                sentiment[i]=1
       
        
    df['Sentiment']=sentiment

    return df

def frequenceVisualization(statements):
    from nltk.probability import FreqDist 
    import matplotlib.pyplot as plt

    fd_pos = FreqDist()
    for statement in statements:
        words = statement.split(" ")
        for word in words:
            if word.lower() not in stops_eng:
                if word.isalpha():
                    fd_pos[word.lower()] += 1
    vocab = fd_pos.keys()
    print(len(vocab))
    print(fd_pos.most_common(20))
    plt.figure(figsize=(16,5))
    fd_pos.plot(20)

def countplot():
    import seaborn as sns
    import matplotlib.pyplot as plt 
    cntplt = sns.countplot(x='Sentiment', data=df)
    cntplt.set_xticklabels(["Positive","Negative","Neutral"])
    plt.show()
    
if __name__ == "__main__": 
         
    df=readData_addSentiment()

    countplot()
    

    # Split Data
    positive_sent_stmts = df.loc[df['Sentiment'] == 2]['Sentence']
    negative_sent_stmts = df.loc[df['Sentiment'] == 1]['Sentence']
    neutral_sent_stmts = df.loc[df['Sentiment'] == 0]['Sentence']

    # stop_words
    from nltk.corpus import stopwords
    stops_eng = stopwords.words('english')
    stops_eng.remove('no')
    stops_eng.remove("not")

    # Word frequency analysis
    #frequenceVisualization(df['Sentence'])
    #frequenceVisualization(positive_sent_stmts)
    #frequenceVisualization(negative_sent_stmts)
    #frequenceVisualization(neutral_sent_stmts)



