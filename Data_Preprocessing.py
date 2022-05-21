from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from num2words import num2words
import numpy as np
import nltk
nltk.data.path.append("C:/Users/White Walker/Documents/FYP/newwww huraira/newwww/venv/nltk_data")
class PreProcessing():

    def convert_lower_case(self, data):
        return np.char.lower(data)

    def remove_stop_words(self, data):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    def remove_punctuation(self, data):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        return data

    def remove_apostrophe(self, data):
        return np.char.replace(data, "'", "")

    def stemming(self, data):
        stemmer = PorterStemmer()

        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    def convert_numbers(self, data):
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    def preprocess(self, data):
        data = self.convert_lower_case(data)
        #print("Lower case data = ", data)
        data = self.remove_punctuation(data)  # remove comma seperately
        #print("Remove punctuation =", data)
        data = self.remove_apostrophe(data)
        #print("Remove apostrophe", data)

        # data = remove_stop_words(data)
        # print("Remove stop words" , data)
        data = self.convert_numbers(data)
        #print("Convert Numbers ", data)
        data = self.stemming(data)
        #print("Stemming", data)
        data = self.remove_punctuation(data)
        data = self.convert_numbers(data)
        data = self.stemming(data)  # needed again as we need to stem the words
        # needed again as num2word is giving few hypens and commas fourty-one
        data = self.remove_punctuation(data)
        # needed again as num2word is giving stop words 101 - one hundred and one
        data = self.remove_stop_words(data)
        return data
