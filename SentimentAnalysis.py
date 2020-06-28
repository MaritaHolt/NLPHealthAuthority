#import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from Dataanalysis import readData_addSentiment

df=readData_addSentiment()

# stop_words
stops_eng = stopwords.words('english')
stops_eng.remove('no')
stops_eng.remove("not")

# Extract relevant data
statements = df["Sentence"]
labels = df["Sentiment"]

# Choice for Text vectorization
tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stops_eng, lowercase=True)  
countvectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=stops_eng, lowercase=True)  

vectorizers =[tfidfvectorizer, countvectorizer]

# Choice of Classifiers
classifiers = [
    #KNeighborsClassifier(5),
    SVC(), 
    #DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    MLPClassifier(),
    MultinomialNB(),
    #LogisticRegression()
    ]
for vectorizer in vectorizers:
    print ("*"*100)
    print(vectorizer.__class__.__name__)
    for clf in classifiers:
        name = clf.__class__.__name__
        name_short = clf.__class__.__name__[:3]
        print("="*30)
        print(name)

        pipeline= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
        crossValidation = True
        # Fit model and predict on test data
        t0 = time.process_time()
        if crossValidation == True:
            score = cross_val_score(pipeline, statements, labels,cv=5, scoring='f1_weighted')
            print("Score:", score, "Mean:", score.mean())
        else:
            # Split in train and test
            stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.2, random_state=30)

            pipeline.fit(stmts_train, labels_train)
            predictions = pipeline.predict(stmts_test)

            # Result visualization
            cf = confusion_matrix(labels_test,predictions)
            print(cf)
            #sns.heatmap(cf, cmap="GnBu", annot=True, fmt='g')
            #plt.show()
            print(classification_report(labels_test,predictions))

        t1 = time.process_time()
        training_time = t1-t0
        print(training_time)





    