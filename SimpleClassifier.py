import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from Dataanalysis import readData_addSentiment

df=readData_addSentiment()
# Extract relevant data
statements = df["clean_text"]
labels = df["Sentiment"]

predictions=np.zeros((labels.shape), dtype='f')

cf = confusion_matrix(labels,predictions)
sns_plot = sns.heatmap(cf, cmap="Blues", annot=True, fmt='g')
sns_plot.get_figure().savefig("Reports/Heatmap_SimpleClassifier.png")
plt.clf()
results=open("Reports/Results_SimpleClassifier.txt", 'w')
results.write(classification_report(labels,predictions))
