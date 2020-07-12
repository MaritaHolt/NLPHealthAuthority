import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

def fit_NB(statements, labels):
    vectorizer=CountVectorizer(ngram_range=(1,2), max_df=0.3)
    clf=MultinomialNB(fit_prior=True, alpha=1.3)
    pipeline_NB= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
    pipeline_NB.fit(statements,labels)
    score = pipeline_NB.score(statements, labels)
    return pipeline_NB, score


def fit_LinSVC(statements, labels):
    vectorizer=CountVectorizer(ngram_range=(1,2), max_df=0.5)
    clf=LinearSVC(C=1.0, class_weight='balanced')
    pipeline_svm= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
    pipeline_svm.fit(statements,labels)
    score = pipeline_svm.score(statements, labels)
    return pipeline_svm, score

def fit_Logreg(statements, labels):
    vectorizer=CountVectorizer(ngram_range=(1,2), max_df=0.5)
    clf=LogisticRegression(C=1.0, class_weight='balanced')
    pipeline_lr= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
    pipeline_lr.fit(statements,labels)
    score = pipeline_lr.score(statements, labels)
    return pipeline_lr, score

# Compare the results of all three classifiers and classify a sentence as 3 (dummy class) if all three classifiers disagree
def compare(df_pred):
    
    pred=np.zeros((df_pred.shape[0],1), dtype='i')
    for i in range(0,len(pred)):
        if df_pred['NB'][i]==df_pred['SVM'][i]:
            pred[i]=df_pred['NB'][i]
        elif df_pred['SVM'][i] == df_pred['LR'][i]:
            pred[i]=df_pred['SVM'][i]
        elif df_pred['NB'][i] == df_pred['LR'][i]:
            pred[i]=df_pred['NB'][i] 
        else:   
            pred[i]=3
       
        
    df_pred['Pred']=pred

    return df_pred

# Analyze the probability results of the NB classifier and mark a sentence if the highest probability is less than 50%
def compare2(df_pred, pred):
    proba_pred=np.zeros((df_pred.shape[0],1), dtype='i')
    for i in range(0,len(pred)):
        if max(pred[i])<0.5:
            proba_pred[i]=3
        else:
            proba_pred[i]=df_pred['NB'][i]

    df_pred['Proba']=proba_pred

    return df_pred




if __name__=='__main__':
    # Read data
    from Dataanalysis import readData_addSentiment, CleanText
    df=readData_addSentiment()


    # Set directory for saving
    str1='Reports/'
    results=open(str1+"Results_SentAna_newdata.txt","w")
    
    # Extract relevant data
    stmts_train = df["clean_text"]
    labels_train = df["Sentiment"]

        
    # Fit models on whole data set
    pip_nb, score_nb = fit_NB(stmts_train, labels_train)
    pip_svm, score_svm = fit_LinSVC(stmts_train, labels_train)
    pip_lr, score_lr = fit_Logreg(stmts_train, labels_train)

    # Some Sentences extracted from EPAR on Xarelto without prior label
    data_test = {'Sentence' : ["Gender, race and weight have none or only a small effect on rivaroxaban AUC.", 
    "the use in patients with creatinine clearance <15 ml/min is not recommended.", 
    "This is in accordance with the normal activity profile of a direct FXa inhibitor.",
    "No clear dose-efficacy response relationship could be established.",
    "There were not many exclusion criteria, which is thought to be advantageous.",
    "In summary the over-all design of the pivotal studies is believed to be acceptable and similar to the pivotal studies on which the approval of other prophylactic agents within this area have been based.",
    "All studies had a double-blind, double-dummy design.",
    "No clinical studies were performed in children."]}
    
    df_test = pd.DataFrame(data=data_test)
    ct = CleanText()
    stmts_test = ct.fit_transform(df_test['Sentence'])
       
    # Predict on test set
    prediction_1=pip_nb.predict(stmts_test)
    prediction_1_proba=pip_nb.predict_proba(stmts_test)
    prediction_2=pip_svm.predict(stmts_test)
    prediction_3=pip_lr.predict(stmts_test)
        
    data = {'NB' : prediction_1, 'SVM' : prediction_2, 'LR' : prediction_3}
    df_pred=pd.DataFrame(data=data)
                   
    # Two different approaches for unclear instances
    df_pred = compare(df_pred)
    df_pred = compare2(df_pred, prediction_1_proba)
     
   
    results.write(df_pred.to_string())
     

    
    












    