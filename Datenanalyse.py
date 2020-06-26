import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
#from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC


# Read data
df=pd.read_excel('../sentences_with_sentiment.xlsx')
df=df.drop(columns=['ID'])

# Add an additional column Sentiment (Positive== 1, Negative == -1, Neutral == 0)
sentiment=np.zeros((df.shape[0],1), dtype='i')
for i in range(0,len(sentiment)):
    if df['Positive'][i]==1:
        sentiment[i]=2
    if df['Negative'][i]==1:
            sentiment[i]=1
    
df['Sentiment']=sentiment

# Data Visualization
#cntplt = sns.countplot(x='Sentiment', data=df)
#cntplt.set_xticklabels(["Negative","Neutral","Positive"])
#plt.show()

# Split Data
positive_sent_stmts = df.loc[df['Sentiment'] == 1]['Sentence']
negative_sent_stmts = df.loc[df['Sentiment'] == -1]['Sentence']
neutral_Sent_stmts = df.loc[df['Sentiment'] == 0]['Sentence']

# stop_words
stops_eng = stopwords.words('english')
stops_eng.remove('no')
stops_eng.remove("not")

# Extract relevant data and split in train and test
statements = df["Sentence"]
labels = df["Sentiment"]
stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.2, random_state=42)



#Transform data
#vectorizer = TfidfVectorizer()  
#features_train = vectorizer.fit_transform(stmts_train).toarray()
#features_test = vectorizer.transform(stmts_test).toarray() 

pipeline_tfidf = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 3), max_features=2500, stop_words=stops_eng, lowercase=True)),
                    	('classifier', SVC())])

# Fit model and predict on test data
pipeline_tfidf.fit(stmts_train, labels_train)
predictions = pipeline_tfidf.predict(stmts_test)

# Result visualization
cf = confusion_matrix(labels_test,predictions)
print(cf)
sns.heatmap(cf, cmap="GnBu", annot=True, fmt='g')
plt.show()
print(classification_report(labels_test,predictions))