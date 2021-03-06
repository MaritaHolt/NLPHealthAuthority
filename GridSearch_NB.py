import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.pipeline import Pipeline

def gridsearch_NB(stmts_train, stmts_test, labels_train, labels_test, score):
    vectorizer = CountVectorizer() 
    clf = MultinomialNB()
    pipeline= Pipeline(steps=[('vec', vectorizer),('clf', clf)])

    param_grid = {
        'clf__alpha': np.linspace(0.5, 1.5, 6),
        'clf__fit_prior': [True, False],
        'vec__ngram_range': [(1,1),(1,2),(1,3),(2,2),(2,3)],
        'vec__max_df': [0.1,0.2,0.3,0.5,0.7],
        'vec__max_features': [None, 1000, 2000, 500]
    }

    
    gscv = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring=score)
    gscv.fit(stmts_train,labels_train)
    
    predictions=gscv.predict(stmts_test)
    
    return gscv.best_score_, gscv.best_params_, accuracy_score(labels_test,predictions), f1_score(labels_test, predictions, average='weighted')
        
   



if __name__=='__main__':
    # Set directory for saving
    str1='Reports/'
    # Store results to txt
    file_results=open(str1+"Results_GridSearch_NB.txt","w")
    # Read data
    from Dataanalysis import readData_addSentiment
    df=readData_addSentiment()
    
    col_names = ['res_gscv', 'test_acc', 'test_f1', 'param']
    for score in ['accuracy', 'f1_weighted']:
        file_results.write(score+"\n")
        results=[]
        for k in range(0,10):
            df=shuffle(df)
            # Extract relevant data
            statements = df["clean_text"]
            labels = df["Sentiment"]

            stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.1, random_state=0)
            
            scoregscv, param, test_acc, test_f1 = gridsearch_NB(stmts_train, stmts_test, labels_train, labels_test, score)
            results.append([ scoregscv,test_acc,test_f1, param])

            
        
        result=pd.DataFrame(results, columns=col_names)
        file_results.write(result.to_string()+ "\n")
        




    