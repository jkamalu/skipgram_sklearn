# author: John Kamalu (jkamalu)
# citation: https://github.com/scikit-learn/scikit-learn/issues/6983#issuecomment-326542149

import re
import scipy
import numpy
from sklearn.feature_extraction.text import CountVectorizer

class SkipGramVectorizer(CountVectorizer):

	def __init__(self, window_size=1, skip_range=(0,0), raw_text=False, split_pattern=r"[\s]+"):

		self.window_size = window_size
		self.skip_range = skip_range
		self.raw_text = raw_text
		self.split_pattern = split_pattern
		self._assert_args()

		super(SkipGramVectorizer, self).__init__()

	def _assert_args(self):
		# window_size assertions
		assert type(self.window_size) == type(int())

		# skip_range assertions
		assert type(self.skip_range) == type(tuple()) 
		assert len(self.skip_range) == 2
		assert self.skip_range[1] >= self.skip_range[0]
		assert self.skip_range[0] >= 0

		# raw_text assertions
		assert type(self.raw_text) == type(bool())

		# split_pattern assertions
		assert type(self.split_pattern) == type(str())

		# shared assertions
		assert self.skip_range[1] < self.window_size		

	def _skip_grams(self, tokens):
		grams = []
		for i in range(len(tokens)-(self.window_size-1)):
			beg_l = i
			end_l = i+self.skip_range[0]
			beg_r = i+self.skip_range[1]
			end_r = i+self.window_size
			gram = tuple(tokens[beg_l:end_l] + tokens[beg_r:end_r])
			grams.append(gram)
		return grams	

	def build_analyzer(self):
		preprocessor = self.build_preprocessor()
		tokenizer = self.build_tokenizer()

		def handle_input(fit_input):
			if not self.raw_text:
				with open(fit_input, "r") as in_file:
					fit_input = in_file.read()
			return self._skip_grams(tokenizer(preprocessor(fit_input)))

		return handle_input 

	def build_tokenizer(self):
		return lambda doc: re.split(self.split_pattern, doc)

	def build_preprocessor(self):
		return lambda doc: doc.lower().strip()

if __name__ == "__main__":

	skv = SkipGramVectorizer(raw_text=False, window_size=5, skip_range=(2, 3), split_pattern=r"[\s]+")
	skv.fit(["test.txt"])
	print(skv.get_feature_names())