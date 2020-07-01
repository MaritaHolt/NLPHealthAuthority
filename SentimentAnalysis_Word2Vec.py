import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def sentimentanalysis_word2vec(statements, labels, scoringparameter):

    
    # Choice for Text vectorization
    class word2vecVectorizer(object):
        def __init__(self):
            self.model =  gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

        def transform(self, raw_documents, copy=True):
            embedding_features = []
            for sent in raw_documents:
                sent = [x for x in sent.split(' ')]
                for i in range(len(sent)):
                    sent[i] = self.model[sent[i]] if sent[i] in self.model.vocab else np.zeros(300)
                embedding_features.append(np.array(np.mean(sent, axis=0)))

            return np.asmatrix(embedding_features)

        def fit(self, raw_documents,y=None):
            return self
     
    vectorizer=word2vecVectorizer()

    scorelist=[]
    training_time=[]
    # Choice of Classifiers
    classifiers = [
        #KNeighborsClassifier(5),
        LinearSVC(), 
        RandomForestClassifier(n_estimators=100, max_depth=3),
        MLPClassifier(solver='lbfgs'),
        LogisticRegression()
        ]

    for clf in classifiers:
        print(clf.__class__.__name__)
        pipeline= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
            
        # Fit model and predict on test data
        t0 = time.process_time()
        score = cross_val_score(pipeline, statements, labels,cv=5, scoring=scoringparameter)
        scorelist.append(score.mean())
        t1 = time.process_time()
        training_time_clf = t1-t0
        training_time.append(training_time_clf)

    return scorelist, training_time

if __name__=='__main__':
    # Read data
    from Dataanalysis import readData_addSentiment
    df=readData_addSentiment()

    
    scoringparameter='accuracy'

    # Set directory for saving
    str1='Reports/'
    file_results=open(str1+"Results_SentAna_W2Vec_"+scoringparameter+".txt","w")

    col_names=["LinSVC", "RF", "MLP", "LogReg"] #"kNN", 
    scores=[]
    traintime=[]
    for k in range(0,1):
        df=shuffle(df)
        # Extract relevant data
        statements = df["clean_text"]
        labels = df["Sentiment"]


        scorelist, training_time = sentimentanalysis_word2vec(statements, labels, scoringparameter)
        scores.append(scorelist)
        traintime.append(training_time)

    results=pd.DataFrame(scores, columns=col_names)
    results.loc['mean'] = results.mean()

    dftt=pd.DataFrame(traintime, columns=col_names)
    dftt.loc['mean'] = dftt.mean()

    file_results.write(results.to_string()+ "\n")
    file_results.write(dftt.to_string()+ "\n")







    