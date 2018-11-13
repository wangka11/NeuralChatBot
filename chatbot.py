from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import math
import operator

from torch.autograd import Variable
import numpy as np
import os
import os.path
import csv


#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataFolder = "data/"

class CornellMovieCorpus:
    def __init__(self, outfile="formatted_lines.txt"):
        self.corpusFolder = dataFolder + "cornell movie-dialogs corpus/"
        self.outfile = self.corpusFolder + outfile
        self.fields = {"lines": ["lineId", "characterId", "movieId", "characterName", "utterance"],
                       "conversations": ["characterId1", "characterId2", "movieId", "utteranceIds", "lines"]}

    def getLines(self):
        lines = {}
        file = self.corpusFolder + "movie_lines.txt"
        fields = self.fields["lines"]
        with open(file, 'r', encoding="iso-8859-1") as f:
            for line in f:
                tokens = line.split(" +++$+++ ")
                lineFields = dict(zip(fields, tokens))
                lines[tokens[0]] = lineFields
        return lines

    def getConversations(self, lines=None):
        conversations = []
        file = self.corpusFolder + "movie_conversations.txt"
        lines = self.getLines() if lines is None else lines
        with open(file, 'r', encoding="iso-8859-1") as f:
            for line in f:
                tokens = line.split(" +++$+++ ")
                tokens[-1] = eval(tokens[-1])
                ids = tokens[-1]
                for i in range(len(ids) - 1):
                    input = lines[ids[i]]['utterance'].strip()
                    output = lines[ids[i + 1]]['utterance'].strip()
                    if input and output:
                        conversations.append([input, output])

                # length = len(ids) if len(ids) % 2 == 0 else len(ids) - 1
                # i = 0
                # while i < length:
                #     input = lines[ids[i]]['utterance'].strip()
                #     output = lines[ids[i+1]]['utterance'].strip()
                #     conversations.append([input, output])
                #     i += 2
        return conversations


    def processData(self):
        if os.path.isfile(self.outfile):
            pass
        else:
            conversations = self.getConversations()
            with open(self.outfile, 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter="\t")
                for conv in conversations:
                    writer.writerow(conv)




SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15
#BATCH_SIZE = 10


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, PAD_index: PAD_token}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1




if __name__ == "__main__":
    corpus = CornellMovieCorpus()
    corpus.processData()