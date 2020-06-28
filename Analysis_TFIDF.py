#import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
#from sklearn import datasets, svm, tree, metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #log_loss, 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from Dataanalysis import readData_addSentiment

df=readData_addSentiment()

# stop_words
stops_eng = stopwords.words('english')
stops_eng.remove('no')
stops_eng.remove("not")

# Extract relevant data and split in train and test
statements = df["Sentence"]
labels = df["Sentiment"]
stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.2, random_state=30)

#Transform data
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stops_eng, lowercase=True)  
X_train = vectorizer.fit_transform(stmts_train).toarray()
#print(vectorizer.get_feature_names())
X_test = vectorizer.transform(stmts_test).toarray() 



classifiers = [
    #KNeighborsClassifier(5),
    SVC(), 
    #DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    MLPClassifier(),
    GaussianNB(),
    #LogisticRegression()
    ]

for clf in classifiers:
    t0 = time.process_time()
    clf.fit(X_train, labels_train)
    t1 = time.process_time()
    training_time = t1-t0
    
    name = clf.__class__.__name__
    name_short = clf.__class__.__name__[:3]
    
    print("="*30)
    print(name)
    
    t0 = time.process_time()
    predictions = clf.predict(X_test)
    predictions_train = clf.predict(X_train)
    t1 = time.process_time()
    predict_time = t1-t0
    print("Train time:", training_time)
    print("Predict time:", predict_time)

    acc_train = accuracy_score(labels_train, predictions_train)
    print("Accuracy on Training Set: {:.4%}".format(acc_train))
    acc = accuracy_score(labels_test, predictions)
    print("Accuracy on Test Set: {:.4%}".format(acc))



#pipeline_tfidf = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 3), max_features=2500, stop_words=stops_eng, lowercase=True)),
#                   	('classifier', SVC())])

# Fit model and predict on test data
#pipeline_tfidf.fit(stmts_train, labels_train)
#predictions = pipeline_tfidf.predict(stmts_test)

    # Result visualization
    cf = confusion_matrix(labels_test,predictions)
    print(cf)
    sns.heatmap(cf, cmap="GnBu", annot=True, fmt='g')
    #plt.show()
    print(classification_report(labels_test,predictions))