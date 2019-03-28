import gensim
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import os
import pickle as pkl
import nltk
from nltk.cluster import KMeansClusterer
from sklearn.neural_network import MLPClassifier

ps = PorterStemmer()

if __name__ == '__main__':
    main()

def tokenize(corpus):
    for line in corpus:
        l = str.encode(line)
        yield gensim.utils.simple_preprocess(l)

def streamming(corpus):
    for d in corpus:
        yield [ps.stem(w) for w in d]

def main(force_train=False, force_kmeans=False):
    link_u_imbd = 'datasets/imdb-unlabeled.txt'
    link_train = 'datasets/train.txt'
    link_test = 'datasets/test.txt'
    df_train = pd.read_csv(link_train, encoding='utf-8')
	df_test = pd.read_csv(link_test, encoding='utf-8')
	df_u_imbd = pd.read_csv(link_u_imbd, encoding='utf-8')
	df_train['label'] = df_train['label'].replace({'neg':'0', 'pos':'1'})
	df_test['label'] = df_test['label'].replace({'neg':'0', 'pos':'1'})


	train_data = df_train['review'].tolist()
	test_data = df_test['review'].tolist()
	unlabeled_data = df_u_imbd['review'].tolist()
	train_y = df_train['label'].tolist()
	test_y = df_test['label'].tolist()

	model = None
	x_train = []

	if force_train or not os.path.isfile('w2v.model'):
		x_train = list(streamming(list(tokenize(train_data))))
		model = gensim.models.Word2Vec(x_train, size=100, window=10, min_count=5)
		model.train(x_train, total_examples=len(x_train), epochs=10)
		model.save('w2v.model')
		pkl.dump(x_train, open("x_train.t", "wb"))
	else:
		model = gensim.models.Word2Vec.load(w2v.model)
		x_train = pkl.load(open("x_train.t", "rb"))


	X = model[model.wv.vocab]
	kcluster = None
	if force_kmeans:
		NUM_CLUSTERS = 2500
		kcluster = KMeansClusterer(NUM_CLUSTERS,  distance=nltk.cluster.util.cosine_distance, repeats=25)
		pkl.dump(kcluster, open("kcluster.t", "wb"))
	else:
		kcluster = pkl.load(open("kcluster.t", "rb"))		

	assigned_clusters = kcluster.cluster(X, assign_clusters=True)
	

	X_train = []
	Y_train = []
	for x in range(len(x_train)):
	    vetores = []
	    for w in x_train[x]:
	        if w not in model.wv.vocab:
	            continue
	        vetor = model.wv[w]
	        vetores.append(np.array(vetor))
	    vec_final = np.mean(vetores, axis=0)
	    X_train.append(vec_final)
	    Y_train.append(train_y[x])
	

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, Y_train)

	
	










