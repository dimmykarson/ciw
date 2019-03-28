import gensim
from nltk.stem import PorterStemmer


ps = PorterStemmer()

def tokenize(corpus):
    for line in corpus:
        l = str.encode(line)
        yield gensim.utils.simple_preprocess(l)

def streamming(corpus):
    for d in corpus:
        yield [ps.stem(w) for w in d]