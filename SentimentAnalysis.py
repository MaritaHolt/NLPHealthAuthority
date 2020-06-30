import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

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

from Dataanalysis import readData_addSentiment

df=readData_addSentiment()


# Extract relevant data
statements = df["clean_text"]
labels = df["Sentiment"]

# Choice for Text vectorization
tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 3))  
countvectorizer = CountVectorizer(ngram_range=(1, 3))  

vectorizers =[tfidfvectorizer, countvectorizer]

# Choice of Classifiers
classifiers = [
    #KNeighborsClassifier(5),
    LinearSVC(), 
    #RandomForestClassifier(),
    MLPClassifier(solver='lbfgs'),
    MultinomialNB(),
    ComplementNB(),
    LogisticRegression()
    ]

# Store results to txt
file_results=open("Results_SentAna_BOW.txt","w")

# Set cross_validation
crossValidation = True

# Test different combinations of vectorizer + classifier    
for vectorizer in vectorizers:
    file_results.write("*"*100+"\n")
    file_results.write(vectorizer.__class__.__name__ +"\n")
    for clf in classifiers:
        name = clf.__class__.__name__
        name_short = clf.__class__.__name__[:3]
        file_results.write("="*30 +"\n")
        file_results.write(name +"\n")

        pipeline= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
        
        # Fit model and predict on test data
        t0 = time.process_time()
        if crossValidation == True:
            score = cross_val_score(pipeline, statements, labels,cv=5, scoring='accuracy')
            file_results.write("Score: " + np.array2string(score) + ", Mean: " + np.array2string(score.mean())+ "\n")
        else:
            # Split in train and test
            stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.2, random_state=20)

            pipeline.fit(stmts_train, labels_train)
           
            predictions=pipeline.predict(stmts_test)
            print(predictions)
            # Result visualization
            cf = confusion_matrix(labels_test,predictions)
            file_results.write(np.array2string(cf, separator=', ')+"\n")
            sns_plot = sns.heatmap(cf, cmap="GnBu", annot=True, fmt='g')
            sns_plot.get_figure().savefig("Heatmap"+vectorizer.__class__.__name__ +clf.__class__.__name__+".png")
            plt.clf()
            clf_report=classification_report(labels_test,predictions, output_dict=True)
            file_results.write(pd.DataFrame(clf_report).transpose().to_string()+ "\n")
            #df_clf_rp=pd.DataFrame(clf_report).transpose().to_latex()

        t1 = time.process_time()
        training_time = t1-t0
        file_results.write("trainging time: " + str(training_time) +"\n")










    