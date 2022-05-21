import numpy
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF():


    def Calculate_tfidf(self):
        filename = "Processed_data.txt"
        with open(filename, encoding="utf8") as my_file:
            data = my_file.read()
        Tfidf_vect_data = TfidfVectorizer(max_features=40)
        dataa = numpy.ravel(data)
        Tfidf_vect_data.fit(dataa)
        Test_X_Tfidf = Tfidf_vect_data.transform(dataa)
        return Test_X_Tfidf





