# NLPHealthAuthority
Collection of Code for a sentiment analysis of EPARs based on single sentences

# Datananalysis
- Read Data from table
- Clean Text (drop duplicates, remove punctation and digits, remove stop words, apply PorterStemmer)
- Different options for data visualization (Countplot, Frequencycount)

# Simple Classifier
Naive Model which labels everything as Positive 

# SentimentAnalysis_BOW
Overwiew over the performance of different models
- Text vectorization via TfidfVectorizer or CountVectorizer
- Comparison of different models (SVM, kNearestNeighbours, MLP, RandomForest, MultinomialNB/ComplementNB, LogisticRegression)

# SentimentAnalysis_Word2Vec
- Comparison of TfidfVectorizer/CountVectorizer with Vectorizer based on Google's word2vec
- Requires Google's pretrained list of words available at https://drive.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
- > Results are not better than with simple BOW. These vectorizer is thus thus not pursued any further 

# Grid Search files
Try to improve most promising models (LinearSVC, MultinomialNB, Logistic Regression) + CountVectorizer via GridSearch

# Final Model
- A combination of all three classifiers with improved Hyperparameters
- Notification in case of unclear classification results possible
# Test on Xarelto EPAR
- Sentences taken from the EPAR on Xarelto are classified after all classifiers are trained on the whole data set
-> The results for negative sentences are poor. 
# Reports
Folder containing data produced by various experiments

