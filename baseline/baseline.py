"""
A baseline method to classify sarcastic tweets

"""
import numpy as np

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

	def test(self, data, classname):
		#data = np.load(filename)
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
		print "precision: ", precision

		# recall, how many correct items are selected
		print "recall: ", recall
		# f-score: 2 * (precision * recall) / (precision + recall)

		print "f: ", f


if __name__ == "__main__":
	model = NBModel("postrain.npy", "negtrain.npy")
	model.train(0.09)
	model.get_stat()
