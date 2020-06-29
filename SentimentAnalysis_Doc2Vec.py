import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


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
class doc2vecVectorizer(object):
    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        
        
    def fit(self, raw_documents, y=None):
        tagged_data = [TaggedDocument(words=word_tokenize(sent.lower()), tags=[str(i)]) for i, sent in enumerate(raw_documents)]
        model = Doc2Vec(documents=tagged_data, vector_size=self.vector_size) 
        for epoch in range(self.epochs):
            model.train(tagged_data, total_examples=len(tagged_data), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha 
            
        self._model = model
        return self
        
    def transform(self, raw_documents, copy=True):
        return np.asmatrix(np.array([self._model.infer_vector(word_tokenize(sent.lower())) for i, sent in enumerate(raw_documents)]))
	


class word2vecVectorizer(object):
    def __init__(self):
        self.model =  gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

    def transform(self, raw_documents, copy=True):
        embedding_features = []
        for sent in raw_documents:
            sent = [x.lower() for x in sent.split(' ') if x not in stops_eng]
            for i in range(len(sent)):
                sent[i] = self.model[sent[i]] if sent[i] in self.model.vocab else np.zeros(300)
            embedding_features.append(np.array(np.mean(sent, axis=0)))

        return np.asmatrix(embedding_features)

    def fit(self, raw_documents,y=None):
        return self
        

#doc2vecvectorizer=doc2vecVectorizer(vector_size=300)
word2vecvectorizer=word2vecVectorizer()
tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=stops_eng,  lowercase=True)  
countvectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=stops_eng,  lowercase=True)  


feature_union = ('feature_union', FeatureUnion([
	('word2vec', word2vecvectorizer),
	('count', countvectorizer),
]))

vectorizers =[('vectorizer', word2vecvectorizer)] #, ('vectorizer', tfidfvectorizer) , ('vectorizer', countvectorizer),feature_union] 


# Choice of Classifiers
classifiers = [
    #KNeighborsClassifier(5),
    #SVC(), 
    #DecisionTreeClassifier(),
    #RandomForestClassifier(n_estimators=100),
    #MLPClassifier(),
    #MultinomialNB(),
    LogisticRegression()
    ]




# Test different combinations of vectorizer + classifier    
for vectorizer in vectorizers:
    print ("*"*100)
    print(vectorizer[1].__class__.__name__)
    for clf in classifiers:
        name = clf.__class__.__name__
        name_short = clf.__class__.__name__[:3]
        print("="*30)
        print(name)

        pipeline= Pipeline(steps=[vectorizer,('classifier', clf)])
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










    