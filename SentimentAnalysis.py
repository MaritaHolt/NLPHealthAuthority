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
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline



def sentimentanalysis_bow(statements, labels, scoringparameter, vectorizer):
    
    # Choice for Text vectorization
    if vectorizer=='tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
    else: 
        vectorizer = CountVectorizer(ngram_range=(1, 2))  

    

    # Choice of Classifiers
    classifiers = [
        KNeighborsClassifier(5),
        LinearSVC(), 
        RandomForestClassifier(),
        MLPClassifier(solver='lbfgs'),
        MultinomialNB(),
        ComplementNB(),
        LogisticRegression()
        ]
   
    scorelist=[]
    training_time=[]

    for clf in classifiers:
       

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

    vectorizer='tfidf'
    scoringparameter='accuracy'

    # Set directory for saving
    str1='Reports/'
    file_results=open(str1+"Results_SentAna_BOW_tfidf_"+scoringparameter+".txt","w")

    col_names=["kNN", "LinSVC", "RF", "MLP", "MNB", "CNB", "LogReg"]
    scores=[]
    traintime=[]
    for k in range(0,11):
        df=shuffle(df)
        # Extract relevant data
        statements = df["clean_text"]
        labels = df["Sentiment"]


        scorelist, training_time = sentimentanalysis_bow(statements, labels, scoringparameter, vectorizer)
        scores.append(scorelist)
        traintime.append(training_time)

    results=pd.DataFrame(scores, columns=col_names)
    results.loc['mean'] = results.mean()

    dftt=pd.DataFrame(traintime, columns=col_names)
    dftt.loc['mean'] = dftt.mean()

    file_results.write(results.to_string()+ "\n")
    file_results.write(dftt.to_string()+ "\n")
    












    