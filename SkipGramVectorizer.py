# author: John Kamalu (jkamalu)
# citation: https://github.com/scikit-learn/scikit-learn/issues/6983#issuecomment-326542149

import re
import scipy
import numpy
from sklearn.feature_extraction.text import CountVectorizer

class SkipGramVectorizer(CountVectorizer):

	def __init__(self, **kwargs):

		self.window_size = kwargs.pop("window_size", 1)
		self.skip_range = kwargs.pop("skip_range", (0,0))

		super(SkipGramVectorizer, self).__init__(kwargs)
		
		# window_size assertions
		assert type(self.window_size) == type(int())

		# skip_range assertions
		assert type(self.skip_range) == type(tuple()) 
		assert len(self.skip_range) == 2
		assert self.skip_range[1] >= self.skip_range[0]
		assert self.skip_range[0] >= 0

		# shared assertions
		assert self.skip_range[1] < self.window_size

	def build_analyzer(self):
		preprocessor = self.build_preprocessor()
		tokenizer = self.build_tokenizer()
		return lambda doc: self._skip_grams(tokenizer(preprocessor(doc)))

	def build_tokenizer(self):
		return lambda doc: re.split(r"[\s]+", doc)

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

if __name__ == "__main__":

	skv = SkipGramVectorizer(window_size=5, skip_range=(2, 3), token_pattern=r"[^\s]+")
	skv.fit(["Where are we going cotton eyed joe you dumb animal ?"])
	print(skv.get_feature_names())