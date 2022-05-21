import sys
import numpy

from sklearn import svm, model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import Data_Preprocessing
from Data_Preprocessing import PreProcessing
aa = PreProcessing()

import joblib
import numpy
import pandas as pd

numpy.set_printoptions(threshold=sys.maxsize)

from sklearn.preprocessing import LabelEncoder

Corpus = pd.read_csv(r"F:\FYP\newwww\mbti_1.csv",encoding='latin-1')
#Corpus = aa.preprocess(str(Corpus))

#print(Corpus)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['posts'],Corpus['type'],test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['posts'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)
print(predictions_SVM)
print("###################################")
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)