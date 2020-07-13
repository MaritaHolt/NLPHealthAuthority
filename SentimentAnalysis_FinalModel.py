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

    conditions =[df_pred['NB']==df_pred['SVM'], df_pred['SVM'] == df_pred['LR'], df_pred['NB'] == df_pred['LR'] ] 
    choices =[df_pred['NB'], df_pred['SVM'], df_pred['NB']]
    df_pred['Pred']=np.select(conditions, choices, default=3) 
    

    return df_pred

# Analyze the probability results of the NB classifier and mark a sentence if the highest probability is less than 50%
def compare2(df_pred, pred):
    maxpred=[max(proba) for proba in pred]
    maxasarray=np.array(maxpred)
    df_pred['Proba']=np.where(maxasarray <0.5, 3, df_pred['NB'])
    
    return df_pred






if __name__=='__main__':
    # Read data
    from Dataanalysis import readData_addSentiment
    df=readData_addSentiment()


    # Set directory for saving
    str1='Reports/'
    results=open(str1+"Results_SentAna.txt","w")
    prediction_results=open(str1+"Predictions.txt","w")
    accuracy=[]
    acc_proba_nb=[]
    f1_proba_nb=[]
    
    acc_nb=[]
    acc_svm=[]
    acc_lr=[]

    f1 = []
    f1_nb=[]
    f1_svm=[]
    f1_lr=[]
    flag = True
    for k in range(0,400):
        df=shuffle(df)
        # Extract relevant data
        statements = df["clean_text"]
        labels = df["Sentiment"]

        # Split Data
        stmts_train, stmts_test, labels_train, labels_test = train_test_split(statements, labels, test_size=0.2)
        # Fit models
        pip_nb, score_nb = fit_NB(stmts_train, labels_train)
        pip_svm, score_svm = fit_LinSVC(stmts_train, labels_train)
        pip_lr, score_lr = fit_Logreg(stmts_train, labels_train)

        
        # Predict on test set
        prediction_1=pip_nb.predict(stmts_test)
        prediction_1_proba=pip_nb.predict_proba(stmts_test)
        prediction_2=pip_svm.predict(stmts_test)
        prediction_3=pip_lr.predict(stmts_test)
        
        data = {'Label': labels_test, 'NB' : prediction_1, 'SVM' : prediction_2, 'LR' : prediction_3}
        
        df_pred=pd.DataFrame(data=data)
                   
        # Two different approaches for unclear instances
        df_pred = compare(df_pred)
        df_pred = compare2(df_pred, prediction_1_proba)

        
        
        # Evaluate performance
        acc_proba_nb.append(accuracy_score(labels_test, df_pred['Proba']))
        f1_proba_nb.append(f1_score(labels_test, df_pred['Proba'], average='weighted'))

        acc_nb.append(accuracy_score(labels_test, df_pred['NB']))
        f1_nb.append(f1_score(labels_test, df_pred['NB'], average='weighted'))

        acc_svm.append(accuracy_score(labels_test, df_pred['SVM']))
        f1_svm.append(f1_score(labels_test, df_pred['SVM'], average='weighted'))

        acc_lr.append(accuracy_score(labels_test, df_pred['LR']))
        f1_lr.append(f1_score(labels_test, df_pred['LR'], average='weighted'))

        accuracy.append(accuracy_score(labels_test, df_pred['Pred']))
        f1.append(f1_score(labels_test, df_pred['Pred'], average='weighted'))

        
        # only for visualization of interesting settings
        if ((not (df_pred.loc[df_pred['Proba']==3].empty)) and (not (df_pred.loc[df_pred['Pred']==3].empty)) and flag and f1_score(labels_test, df_pred['Pred'], average='weighted')>0.7):
            cf = confusion_matrix(labels_test,df_pred['Pred'])
            sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
            sns_plot.get_figure().savefig("Reports/Heatmap_FinalModel_overall"+str(k)+".png")
            plt.clf()
        
            cf = confusion_matrix(labels_test,df_pred['NB'])
            sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
            sns_plot.get_figure().savefig("Reports/Heatmap_FinalModel_NB"+str(k)+".png")
            plt.clf()

            cf = confusion_matrix(labels_test,df_pred['Proba'])
            sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
            sns_plot.get_figure().savefig("Reports/Heatmap_FinalModel_Proba"+str(k)+".png")
            plt.clf()

            cf = confusion_matrix(labels_test,df_pred['LR'])
            sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
            sns_plot.get_figure().savefig("Reports/Heatmap_FinalModel_LR"+str(k)+".png")
            plt.clf()

            cf = confusion_matrix(labels_test,df_pred['SVM'])
            sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
            sns_plot.get_figure().savefig("Reports/Heatmap_FinalModel_SVM"+str(k)+".png")
            plt.clf()

            prediction_results.write(df_pred.to_string())

            flag = False
       

        

    # Store all results in a DataFrame
    data={'NB acc_proba': acc_proba_nb, 'NB f1_proba': f1_proba_nb,'NB acc' : acc_nb, 'NB f1' : f1_nb, 
    'SVM acc' : acc_svm, 'SVM f1' : f1_svm, 'LR acc' : acc_lr, 'LR f1' : f1_lr,
    'Model acc' : accuracy, 'Model f1' : f1 }
    performance=pd.DataFrame(data=data)
    performance.loc['mean']=performance.mean()
    performance.loc['max'] = performance.max()
    performance.loc['min'] = performance.min()
    performance.loc['median'] = performance.median()
    performance.loc['std'] = performance.std()
    results.write(performance.to_string())
     

    
    












    