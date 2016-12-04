import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib
from nltk.stem import *
from nltk.stem.porter import *


if __name__ == '__main__':

  #train_file = open(sys.argv[1], 'r')
  #test_file = open(sys.argv[2], 'r')

  sizes = []
  f1_scores_nb = []
  f1_scores_svm = []
  f1_scores_lr = []
  f1_scores_rf = []

  train_data = datasets.load_files("Selected 20NewsGroup/Training",decode_error='ignore',encoding='utf-8',shuffle=True)
  test_data = datasets.load_files("Selected 20NewsGroup/Test",decode_error='ignore',encoding='utf-8')
  docs_test = test_data.data

  # Removing header
  for i in range(len(train_data.data)):
    train_data.data[i] = "\n".join(train_data.data[i].split("\n")[3:])

  # Extracting features
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(train_data.data)

  tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  X_train_tf = tf_transformer.transform(X_train_counts)
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

  # Stemming data
  stemmer = PorterStemmer()
  words = []
  st = []
  for i in range(len(train_data.data)):
    words = train_data.data[i].split(" ")
    singles = [stemmer.stem(word) for word in words]
    st.append(' '.join(singles))
  

  # Naive Bayes
  print("Naive Bayes")
  print("\n")
  text_clf_nb = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                    ])
  text_clf_1 = text_clf_nb.fit(st, train_data.target)
  predicted1 = text_clf_1.predict(docs_test)
  print(metrics.classification_report(test_data.target, predicted1, target_names=test_data.target_names))

  # SVM Classifier
  print("SVM Classifier")
  print("\n")
  text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge',penalty='l2'))
                      ])
  text_clf_2 = text_clf_svm.fit(st, train_data.target)
  predicted2 = text_clf_2.predict(docs_test)
  #svm.SVC(kernel='rbf')
  print(metrics.classification_report(test_data.target, predicted2, target_names=test_data.target_names)) 

  #Logistic Regression
  print("Logistic Regression")
  print("\n")
  text_clf_lr = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LogisticRegression()),
                      ])
  text_clf_3 = text_clf_lr.fit(st, train_data.target)
  predicted3 = text_clf_3.predict(docs_test)
  print(metrics.classification_report(test_data.target, predicted3, target_names=test_data.target_names))

  #Random Forest
  print("Random Forest")
  print("\n")
  text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', RandomForestClassifier()),
                      ])
  text_clf_4 = text_clf_rf.fit(st, train_data.target)
  predicted4 = text_clf_4.predict(docs_test)
  print(metrics.classification_report(test_data.target, predicted4, target_names=test_data.target_names))


  # Splitting Training size
  size1 = 0.2 * len(train_data.data)
  sizes.append(size1)
  
  size2 = 0.4 * len(train_data.data)
  sizes.append(size2)

  size3 = 0.6 * len(train_data.data)
  sizes.append(size3)

  size4 = 0.8 * len(train_data.data)
  sizes.append(size4)

  # Loop for different splits in training sets
  for s in sizes:
    
    train = train_data.data[0:int(s)]
    train_target = train_data.target[0:int(s)]
    #Naive Bayes
    text_clf_split_nb = text_clf_nb.fit(train, train_target)
    predicted_nb = text_clf_split_nb.predict(docs_test)
    f1_scores_nb.append(metrics.f1_score(test_data.target, predicted_nb, average='macro'))

    #SVM
    text_clf_split_svm = text_clf_svm.fit(train, train_target)
    predicted_svm = text_clf_split_svm.predict(docs_test)
    f1_scores_svm.append(metrics.f1_score(test_data.target, predicted_svm, average='macro'))

    #Logistic Regression
    text_clf_split_lr = text_clf_lr.fit(train, train_target)
    predicted_lr = text_clf_split_lr.predict(docs_test)
    f1_scores_lr.append(metrics.f1_score(test_data.target, predicted_lr, average='macro'))

    #Random Forest
    text_clf_split_rf = text_clf_rf.fit(train, train_target)
    predicted_rf = text_clf_split_rf.predict(docs_test)
    f1_scores_rf.append(metrics.f1_score(test_data.target, predicted_rf, average='macro'))

  #plt.title("Learning curve for Naive Bayes")
  plt.ylabel("F1-scores")
  plt.xlabel("Training Sizes")
  plt.plot(sizes, f1_scores_nb, label="Naive Bayes")

  #plt.title("Learning curve for SVM")
  plt.ylabel("F1-scores")
  plt.xlabel("Training Sizes")
  plt.plot(sizes, f1_scores_svm, label="SVM")

  #plt.title("Learning curve for Logistic Regression")
  plt.ylabel("F1-scores")
  plt.xlabel("Training Sizes")
  plt.plot(sizes, f1_scores_lr, label="Logistic Regression")

  #plt.title("Learning curve for Random Forest")
  plt.ylabel("F1-scores")
  plt.xlabel("Training Sizes")
  plt.plot(sizes, f1_scores_rf, label="Random Forest")

  plt.grid(True)
  plt.legend(loc='best')
  plt.title("Training Size vs F1-score")
  plt.savefig("Legend plots")
  plt.close()
  
  #Code to dump and load 

  #joblib.dump(text_clf_2, 'classifier.pkl')
  #classifier = joblib.load('classifier.pkl')
  #predicted_temp = classifier.predict(docs_test)
  #print("Loading.........")
  #print(metrics.classification_report(test_data.target, predicted_temp, target_names=test_data.target_names)) 









