import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.pipeline import Pipeline

def gridsearch_lr(stmts_train, stmts_test, labels_train, labels_test, score):
    vectorizer = CountVectorizer() 
    clf = LogisticRegression()
    pipeline= Pipeline(steps=[('vec', vectorizer),('clf', clf)])

    param_grid = {
        'clf__C': [0.1,1,10,100],
        'clf__class_weight': ['balanced', None, {0: 1, 1: 5, 2: 2}],
        'vec__ngram_range': [(1,1),(1,2)],
        'vec__max_df': [0.5]
    }

    
    gscv = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring=score)
    gscv.fit(stmts_train,labels_train)
    
    predictions=gscv.predict(stmts_test)
    return gscv.best_score_, gscv.best_params_, accuracy_score(labels_test,predictions), f1_score(labels_test, predictions, average='weighted')
        
   



if __name__=='__main__':
    # Set directory for saving
    str1='Reports/'
    # Store results to txt
    file_results=open(str1+"Results_GridSearch_LogReg.txt","w")
    # Read data
    from Dataanalysis import readData_addSentiment
    df=readData_addSentiment()
    col_names = ['res_gscv', 'test_acc', 'test_f1', 'param']
    for score in ['accuracy', 'f1_weighted']:
        file_results.write(score+"\n")
        results=[]
        for k in range(0,100):
            df=shuffle(df)
            # Extract relevant data
            statements = df["clean_text"]
            labels = df["Sentiment"]

            stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.1, random_state=0)
            
            scoregscv, param, test_acc, test_f1 = gridsearch_lr(stmts_train, stmts_test, labels_train, labels_test, score)
            results.append([ scoregscv,test_acc,test_f1, param])
        

            
     
        result=pd.DataFrame(results, columns=col_names)
        file_results.write(result.to_string()+ "\n")



    