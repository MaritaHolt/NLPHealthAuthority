import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from IPython.display import display
def fit_NB(statements, labels):
    vectorizer=CountVectorizer(ngram_range=(1,2), max_df=0.3)
    clf=MultinomialNB(fit_prior=True, alpha=1.3)
    pipeline_NB= Pipeline(steps=[('vectorizer', vectorizer),('classifier', clf)])
    pipeline_NB.fit(statements,labels)
    score = pipeline_NB.score(statements, labels)
    return pipeline_NB, score






if __name__=='__main__':
    # Read data
    from Dataanalysis import readData_addSentiment, CleanText
    import eli5
    from eli5.lime import TextExplainer
    df=readData_addSentiment()

    
    # Extract relevant data
    mask=df.index.isin(['98','7','57', '96', '103', '207'])
    stmts_train = df["clean_text"][~mask]
    labels_train = df["Sentiment"][~mask]
    print(len(stmts_train))

        
    # Fit models on whole data set
    pip_nb, score_nb = fit_NB(stmts_train, labels_train)
       
    # Interpretation of some sentences
    data_test = {'Sentence' : ["Poor tolerability of the combination is manifest with high rates of discontinuations due to AEs and dose modifications.",
    "This means that further evidence on this medicinal product is awaited",
    "More detailed data on injection site reactions, hypersensitivity and anaphylactic reactions were requested in order to allow a thorough assessment of this issue both in subjects with and without ADAs.",
    "However, precaution is warranted given a small size of safety database, a limited information on long-term toxicity, and a limited data on PK/PD interactions together with indication on potential for worsening toxicity for VEGFRi -mTOR inhibitor combinations in general.",
    "Study 010 will assess in vitro lenvatinib protein binding, determine the unbound drug concentrations in order to define correctly the dose-adjustment in patients with severe hepatic and renal impairment.",
    "Study EFC12404 provides relevant information on the contribution of the mono-components to the effect of the FRC."]}
    
    df_test = pd.DataFrame(data=data_test)
    ct = CleanText()
    stmts_test = ct.fit_transform(df_test['Sentence'])
       
    # Predict on test set
    prediction_1=pip_nb.predict(stmts_test)
    prediction_1_proba=pip_nb.predict_proba(stmts_test)
    print(prediction_1_proba)
        
     
    # Text interpretation
    class_names=['Positive','Negative','Neutral']
    te = TextExplainer(random_state=42)
    te.fit(stmts_test[5], pip_nb.predict_proba)
    display(te.metrics_)
  
    display(te.show_prediction(target_names=class_names))
    print('finished')
    
        












    