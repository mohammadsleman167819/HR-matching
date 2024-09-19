import re
import nltk
from nltk.stem import PorterStemmer




def preprocess(text):

    final_string = ""
    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', ' ', text)

    # Remove unwanted punctuation
    text = re.sub(r'"', ' ', text)  
    text = re.sub(r"[£'!$%&()*,-./:;<=>?@[\]^_`{|}~]",' ',text)
    text = re.sub(r"\s+",' ',text)
    

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    
    text_filtered = [word for word in text if not word in useless_words]
    stemmer = PorterStemmer() 
    text_stemmed = [stemmer.stem(y) for y in text_filtered]
    

    final_string = ' '.join(text_stemmed)
    return final_string
