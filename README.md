# NLPHealthAuthority
Collection of Code for a sentiment analysis of EPARs based on single sentences

# Datananalysis
- Read Data from table
- Different options for data visualization (Countplot, Frequencycount)

# SentimentAnalysis
Overwiew over the performance of different models
- Text vectorization via TfidfVectorizer or CountVectorizer
- Comparison of different models (SVM, kNearestNeighbours, MLP, RandomForest, MultinomialNB, LogisticRegression)

# SentimentAnalysis_Doc2Vec
Comparison of TfidfVectorizer/CountVectorizer with Vectorizers based on Google's word2vec and doc2vec (including FeatureUnion)
Requires Google's pretrained list of words available at https://drive.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
- > Results are not better than with simple BOW. These vectorizers are thus not pursued any further 

