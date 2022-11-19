# following http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html
# Byte Pair Encoding

import re
from collections import Counter

from ..tensor import Tensor

class Embedding:
  def __call__(self, text):
    vocab = self._get_vocab(text)
    pair_stats = self._get_pair_stats(vocab)
    best_pair = max(pair_stats, key=pair_stats.get) 
    new_vocab = self._merge_vocab(best_pair, vocab)
    print(new_vocab)

  def _get_vocab(self, text):
    return Counter(text.split()).most_common()

  def _get_pair_stats(self, vocab):
    # get counts of pairs of consecutive symbols
    pairs = {}
    for word, freq in vocab:
      for i in range(len(word) - 1):
        pair = (word[i], word[i+1])
        curr_freq = pairs.get(pair, 0)
        pairs[pair] = curr_freq + freq
    return pairs

  def _merge_vocab(self, best_pair, vocab):
    # merge the occurrences of the most frequent pair
    out = {}
    pattern = re.escape(' '.join(best_pair))
    replacement = ''.join(best_pair)
    print(vocab)
    for word in vocab:
      # replace most frequent pair in all vocab
      word_out = re.sub(pattern, replacement, word[0])
      out[word_out] = vocab[word[1]]
    return out