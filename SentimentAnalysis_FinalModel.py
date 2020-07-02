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
    return pipeline_NB


def fit_LinSVC(statements, labels):
    vectorizer=CountVectorizer(ngram_range=(1,2), max_df=0.5)
    clf=LinearSVC(C=1.0, class_weight='balanced')
    pipeline_svm= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
    pipeline_svm.fit(statements,labels)
    return pipeline_svm

def fit_Logreg(statements, labels):
    vectorizer=CountVectorizer(ngram_range=(1,2), max_df=0.5)
    clf=LogisticRegression(C=0.1, class_weight='balanced')
    pipeline_lr= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
    pipeline_lr.fit(statements,labels)
    return pipeline_lr

def predict(statements, pipeline_1, pipeline_2, pipeline_3):
    prediction_1=pipeline_1.predict(statements)
    prediction_2=pipeline_2.predict(statements)
    prediction_3=pipeline_3.predict(statements)
    data = {'NB' : prediction_1, 'SVM' : prediction_2, 'LR' : prediction_3}
    df_pred=pd.DataFrame(data=data)
    
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


if __name__=='__main__':
    # Read data
    from Dataanalysis import readData_addSentiment
    df=readData_addSentiment()


    # Set directory for saving
    str1='Reports/'
    results=open(str1+"Results_SentAna.txt","w")
    Prediction=open(str1+"Predictions_SentAna.txt", "w")
    accuracy=[]
    acc_nb=[]
    acc_svm=[]
    acc_lr=[]
    f1 = []
    f1_nb=[]
    f1_svm=[]
    f1_lr=[]
    for k in range(0,10):
        df=shuffle(df)
        # Extract relevant data
        statements = df["clean_text"]
        labels = df["Sentiment"]


        stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.2, random_state=0)

        pip_nb=fit_NB(stmts_train, labels_train)
        pip_svm=fit_LinSVC(stmts_train, labels_train)
        pip_lr=fit_Logreg(stmts_train, labels_train)
        
        
        df_pred = predict(stmts_test, pip_nb, pip_svm, pip_lr)
        

        acc_nb.append(accuracy_score(labels_test, df_pred['NB']))
        f1_nb.append(f1_score(labels_test, df_pred['NB'], average='weighted'))

        acc_svm.append(accuracy_score(labels_test, df_pred['SVM']))
        f1_svm.append(f1_score(labels_test, df_pred['SVM'], average='weighted'))

        acc_lr.append(accuracy_score(labels_test, df_pred['LR']))
        f1_lr.append(f1_score(labels_test, df_pred['LR'], average='weighted'))

        accuracy.append(accuracy_score(labels_test, df_pred['Pred']))
        f1.append(f1_score(labels_test, df_pred['Pred'], average='weighted'))

       

        cf = confusion_matrix(labels_test,df_pred['Pred'])
        sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
        sns_plot.get_figure().savefig("Reports/Heatmap_FinalModel_overall"+str(k)+".png")
        plt.clf()
       

        


    data={ 'NB acc' : acc_nb, 'NB f1' : f1_nb, 'SVM acc' : acc_svm, 'SVM f1' : f1_svm,
    'LR acc' : acc_lr, 'LR f1' : f1_lr,'Model acc' : accuracy, 'Model f1' : f1 }
    performance=pd.DataFrame(data=data)
    performance.loc['mean']=performance.mean()
    results.write(performance.to_string())
     

    
    












    