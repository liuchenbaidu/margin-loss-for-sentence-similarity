import re
import os
import torch
import itertools
import random
from tqdm import tqdm

random.seed(666)
# Default word tokens
PAD_token = 0  # Used for padding short sentences

MAX_LENGTH = 100  # Maximum sentence length to consider
MIN_COUNT = 5    # Minimum word count threshold for trimming

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1 # Count default tokens

        for word in keep_words:
            self.addWord(word)

def seg_sentence(sentence, model_path = '../public_data/ltp_data_v3.4.0/cws.model'):
    from pyltp import Segmentor
    segmentor = Segmentor()
    segmentor.load(model_path)
    words_list = segmentor.segment(sentence)
    return ' '.join(words_list)

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    
    pairs_ 	= {}
    train_pairs = []
    val_pairs 	= []
    test_pairs 	= []
    
    val_num_for_each_sample = 2
    test_num_for_each_sample = 1

    for pair in pairs:
	    if pair[0] not in pairs_.keys():
		    pairs_[pair[0]] = [pair]
	    else:
		    pairs_[pair[0]].append(pair)
    for key, value in pairs_.items():
	   
	    for i in range(val_num_for_each_sample):
		    val_pairs.append(value[i])
	    for j in range(val_num_for_each_sample, val_num_for_each_sample+test_num_for_each_sample):
		    test_pairs.append(value[j])
	    for k in range(val_num_for_each_sample+test_num_for_each_sample, len(value)):
		    train_pairs.append(value[k])
    
    voc = Voc(corpus_name)
    return voc, pairs, train_pairs, val_pairs, test_pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[1]) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, datafile):
    voc, pairs, train_pairs, val_pairs, test_pairs = readVocs(datafile, corpus_name)
    for elem in tqdm(pairs):
                #voc.addSentence( seg_sentence(elem[1]) )
                voc.addSentence( ' '.join( elem[1] ) )
    print("Counted words:", voc.num_words)
    return voc, train_pairs, val_pairs, test_pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        pair_ = pair[0].split('\t')
        input_sentence = pair_[0]
        output_sentence = pair_[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs



def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(in_, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in in_]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, lengths, max_target_len

def outputVar(out_):
    indexes_batch = [int(out) for out in out_]
    padVar = torch.LongTensor(indexes_batch)
    return padVar

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[1]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(' '.join( pair[1] ))
        output_batch.append(pair[0])
    
    input_, mask, input_len, max_input_len  = inputVar(input_batch, voc)
    output_                                 = outputVar(output_batch)
    return input_, mask, input_len, max_input_len, output_

if __name__ == '__main__':

    corpus_name = 'iask'
    datafile = 'data/data_iask.csv' 
    voc, train_pairs, val_pairs, test_pairs = loadPrepareData(corpus_name, datafile)
    
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(train_pairs) for _ in range(small_batch_size)])
    pass
