"""
A baseline method to classify sarcastic tweets

"""
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models


from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
import string

import multiprocessing
from senti_classifier import senti_classifier

import re

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class TopicExtractor(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		tokenizer = RegexpTokenizer(r'\w+')
		# english stop words
		en_stop = get_stop_words('en')
		p_stemmer = PorterStemmer()
		dictionary = corpora.Dictionary(posts)
		texts = []
		for p in posts:
			tokens = tokenizer.tokenize(p.lower())
			stopped_tokens = [i for i in tokens if not i in en_stop]
			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			texts.append(stemmed_tokens)
		dictionary = corpora.Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)

class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


def getScore(post):
  pos_score, neg_score = senti_classifier.polarity_scores([post])
  #print str(pos_score) + " " + str(neg_score)
  return [pos_score, neg_score]

class CaptilizationExtractor(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		features = [[0] for i in range(len(posts))]
		for i, p in enumerate(posts):
			toks = p.split()
			cap_count = 0
			for t in toks:
				if t.isupper():
					features[i][0] = 1
					break
				if t[0].isupper():
					cap_count += 1
			if cap_count > 3:
				features[i][0] = 1
		return features

class PuncuationExtractor(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.punct = [(re.compile('\!+'), 0), (re.compile('\?+'), 1), (re.compile('\.+'), 2)]

	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		features = [[0 for i in range(len(self.punct))] for i in range(len(posts))]
		for i, p in enumerate(posts):
			tokens = p.split()
			for t in tokens:
				for pattern, value in self.punct:
					if pattern.match(t):
						features[i][value] = 1
		return features

class TopicExtractor(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		pass

class EmotionExtractor(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, posts):
    """ Converts posts into a list of lists of values, where each inner list
        is a 2 element vector with the first element being a positive emotion score
        and the second element being a negative emotion score
    """
    pool = multiprocessing.Pool()
    return pool.map(getScore, posts, 100)

class SVM(object):
	def __init__(self, pos_file, neg_file):
		self.data = []
		for i in range(9):
			pos_file = "pos_data/pos" + str(i) + ".txt"
			with open(pos_file) as myfile:
				for line in myfile:
					self.data.append([line.rstrip(), '1'])
		for i in range(9):
			neg_file = "neg_data/neg" + str(i) + ".txt"
			with open(neg_file) as myfile:
				for line in myfile:
					self.data.append([line.rstrip(), '0'])

		np.random.shuffle(self.data)

		x, y = [], []
		for d in self.data:
			x.append(d[0])
			y.append(int(d[1]))

		self.data = x
		self.y_train = y

		self.pipeline = Pipeline([
			('union', FeatureUnion(
				transformer_list = [
					('bag_words', Pipeline([
						('preprocessor', NLTKPreprocessor()),
						('tfidf', TfidfVectorizer(ngram_range=(1, 2), tokenizer=identity, preprocessor=None, lowercase=False)),
						#('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.5, stop_words='english')),
						('best', TruncatedSVD(n_components=50))
						])),
					# add other features here as an element in transformer list
					('capitalize', Pipeline([
						('cap_words', CaptilizationExtractor())
						])),
					('punctuation', PuncuationExtractor())
          ('emotion', Pipeline([
            ('emotion_words', EmotionExtractor())
            ]))
					]
				)),
			('svc', svm.SVC()),
			])	
	def train(self):
		print ("start training")
		self.pipeline.fit(self.data, self.y_train)
		print ("finish training")

	def test(self, file_name, class_id):
		data = [line.rstrip("\n") for line in open(file_name)]
		y_predict = self.pipeline.predict(data)
		print (sum(y_predict), len(y_predict))


class NBModel(object):
	"""
	bag of words Naive Bayes model
	"""
	def __init__(self, pos_file, neg_file):
		self.k_count = {}
		self.kw_count = {}
		self.k_prob = {}
		self.kw_prob = {}
		self.v_set = set()

		self.NON_EXIST_TOK = "NON_EXIST"
		self.POS_CLASS = "+"
		self.NEG_CLASS = "-"

		self.pos_data = np.load(pos_file)
		self.neg_data = np.load(neg_file)

	def train(self, smooth_factor):
		self.kw_count[self.POS_CLASS] = {}
		self.kw_count[self.NEG_CLASS] = {}
		self.k_count[self.POS_CLASS] = len(self.pos_data)
		self.k_count[self.NEG_CLASS] = len(self.neg_data)
		for sent in self.pos_data:
			tokens = sent.split()
			for w in tokens:
				self.kw_count[self.POS_CLASS][w] = self.kw_count[self.POS_CLASS].get(w, 0) + 1
				self.v_set.add(w)
		for sent in self.neg_data:
			tokens = sent.split()
			for w in tokens:
				self.kw_count[self.NEG_CLASS][w] = self.kw_count[self.NEG_CLASS].get(w, 0) + 1
				self.v_set.add(w)

		v = len(self.v_set)
		for k, count_dic in self.kw_count.items():
			k_sum_wcount = sum(count_dic.values())
			self.kw_prob[k] = {}
			for w, c in count_dic.items():
				self.kw_prob[k][w] = (c + smooth_factor) / (k_sum_wcount + v * smooth_factor)
			self.kw_prob[k][self.NON_EXIST_TOK] = smooth_factor / (k_sum_wcount + v * smooth_factor)

		k_sum_count = sum(self.k_count.values())
		for k, c in self.k_count.items():
			self.k_prob[k] = (c + 0.0) / k_sum_count

	def compute_sentence(self, string):
		tokens = string.split()
		max_p, result = -np.inf, None
		for k in self.k_prob.keys():
			p = np.log(self.k_prob[k])
			for w in tokens:
				if w in self.kw_prob[k]:
					p += np.log(self.kw_prob[k][w])
				else:
					p += np.log(self.kw_prob[k][self.NON_EXIST_TOK])
			#print k, p
			if p > max_p:
				max_p = p
				result = k 
		#print "result: ", result
		return result

	def test(self, filename, classname):
		#data = np.load(filename)
		data = [line.rstrip("\n") for line in open(filename)]
		total, correct = 0, 0
		for d in data:
			r = self.compute_sentence(d)
			if r == classname:
				correct += 1
			total += 1
		return correct, total, float(correct) / total


	def get_stat(self):
		pos_test = np.load("postest.npy")
		neg_test = np.load("negtest.npy")
		totalcount = len(pos_test) + len(neg_test)

		pos_correct, pos_total, pos_rate = self.test(pos_test, "+")
		neg_correct, neg_total, neg_rate = self.test(neg_test, "-")

		precision = pos_rate
		recall = float(pos_correct) / (pos_correct + neg_total - neg_correct)
		f = 2 * precision * recall / (precision + recall)
		# precision, true positive, how many selected items are correct
		print ("precision: ", precision)
		# recall, how many correct items are selected
		print ("recall: ", recall)
		# f-score: 2 * (precision * recall) / (precision + recall)
		print ("f: ", f)


if __name__ == "__main__":
	#model = NBModel("postrain.npy", "negtrain.npy")
	#model.train(0.09)
	#model.get_stat()
	model = SVM("pos_data/pos0.txt", "neg_data/neg0.txt")
	model.train()
	model.test("negtest.txt", 1)
	model.test("postest.txt", 1)
