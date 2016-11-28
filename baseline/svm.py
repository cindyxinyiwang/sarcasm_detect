"""
A baseline method to classify sarcastic tweets

"""
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.cross_validation import train_test_split as tts
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
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


def build_and_evaluate(X, y, classifier=svm.SVC, verbose=True):
	def build(classifier, X, y=None):
		if isinstance(classifier, type):
			classifier = classifier()

		model = Pipeline([
			('union', FeatureUnion(
				transformer_list = [
					('bag_words', Pipeline([
						('preprocessor', NLTKPreprocessor()),
						#('tfidf', TfidfVectorizer(ngram_range=(1, 2), tokenizer=identity, preprocessor=None, lowercase=False)),
						#('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.5, stop_words='english')),
						('topics_and_ngrams', FeatureUnion(transformer_list = [
							('grams', Pipeline([
								('ngram', TfidfVectorizer(ngram_range=(1, 2), tokenizer=identity, preprocessor=None, lowercase=False)),
								('best', TruncatedSVD(n_components=50))
								])),
							#('topics', Pipeline([
							#	('tfid', TfidfVectorizer(ngram_range=(1, 1), tokenizer=identity, preprocessor=None, lowercase=False)),
							#	('topic', NMF(n_components=9, random_state=1,
          					#	alpha=.1, l1_ratio=.5)),
							#	])),			
							])),
						])),
					# add other features here as an element in transformer list
					('capitalize', Pipeline([
						('cap_words', CaptilizationExtractor())
						])),
					('punctuation', PuncuationExtractor())
          #('emotion', Pipeline([
           # ('emotion_words', EmotionExtractor())
            #]))
					]
				)),
			('svc', svm.SVC()),
			])	
		model.fit(X, y)
		return model

	labels = LabelEncoder()
	y = labels.fit_transform(y)

	if verbose: print("Building for evaluation")
	X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
	model = build(classifier, X_train, y_train)

	if verbose:

		print("classification Report: \n")

	y_pred = model.predict(X_test)
	print(clsr(y_test, y_pred))

if __name__ == "__main__":
	data = []
	for i in range(9):
		pos_file = "pos_data/pos" + str(i) + ".txt"
		with open(pos_file) as myfile:
			for line in myfile:
				data.append([line.rstrip(), '1'])
	for i in range(9):
		neg_file = "neg_data/neg" + str(i) + ".txt"
		with open(neg_file) as myfile:
			for line in myfile:
				data.append([line.rstrip(), '0'])
	np.random.shuffle(data)
	x, y = [], []
	for d in data:
		x.append(d[0])
		y.append(int(d[1]))
	build_and_evaluate(x, y)