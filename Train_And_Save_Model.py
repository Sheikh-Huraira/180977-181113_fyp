import joblib
import numpy
import pandas as pd
import sys
import numpy

from sklearn import  svm
from sklearn.feature_extraction.text import TfidfVectorizer

numpy.set_printoptions(threshold=sys.maxsize)

from sklearn.preprocessing import LabelEncoder

Corpus = pd.read_csv(r"F:\FYP\newwww\mbti_1.csv",encoding='latin-1')


Train_X = Corpus['posts']
Train_Y = Corpus['type']

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)

with open('label.txt', 'w') as f:
    f.write(str(Train_Y))

Tfidf_vect = TfidfVectorizer(max_features=40)
Tfidf_vect.fit(Corpus['posts'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)



SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
filename = 'finalized_model.sav'

joblib.dump(SVM, filename)