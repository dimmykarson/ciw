from sklearn.metrics import accuracy_score
import gensim
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import os
import pickle as pkl
import nltk
from nltk.cluster import KMeansClusterer
from sklearn.neural_network import MLPClassifier
from utils import *


if __name__ == '__main__':
    main()


def main(force_retrain=True, dim=100):
    link_u_imbd = 'datasets/imdb-unlabeled.txt'
    link_train = 'datasets/train.txt'
    link_test = 'datasets/test.txt'

    df_test = pd.read_csv(link_test, encoding='utf-8')
    df_test['label'] = df_test['label'].replace({'neg':'0', 'pos':'1'})
    test_data = df_test['review'].tolist()
    test_y = df_test['label'].tolist()

    x_test = list(streamming(list(tokenize(test_data))))
    model = gensim.models.Word2Vec.load("w2v.model")

    if force_retrain:
        print("Force re-train")
        model.build_vocab(x_test, update=True)
        model.train(x_test, total_examples=len(x_test), epochs=10)
        model.save("w2v.rt.model")



    X_test = []
    Y_test = []
    for x in range(len(x_test)):
        vectors = []
        for w in x_test[x]:
            vector = [0]*dim
            if w in model.wv.vocab:
                vector = model.wv[w]
            vectors.append(np.array(vector))
        vec_final = np.mean(vectors, axis=0)
        X_test.append(vec_final)
        Y_test.append(test_y[x])


    clf = pkl.load(open("clf.model", "rb"))
    print("Predict")
    Y_predict = clf.predict(X_test)

    score = accuracy_score(Y_test, Y_predict)






