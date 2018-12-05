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
from http.cookiejar import CookieJar
import urllib.request
import ssl
import certifi
import bs4
import re
from bs4 import BeautifulSoup
import itertools

from torch.autograd import Variable
import numpy as np
import os
import os.path
import csv
from utils import *


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataFolder = "data/"


class TBBTCorpus:
    def __init__(self):
        self.corpusFolder = dataFolder + "the big bang theory/"
        #self.characterList = ["sheldon"]

    def getEpisodes(self):
        home_page = "https://bigbangtrans.wordpress.com/"
        response = urllib.request.urlopen(home_page, cafile=certifi.where())
        content = response.read()
        soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
        episodes = soup.find_all('li', class_='page_item')
        pages = []
        for e in episodes:
            pages.append(e.contents[0]['href'])
        return pages

    def getOutFile(self, character):
        pages = self.getEpisodes()
        outfile = self.corpusFolder + character + ".txt"
        if os.path.isfile(outfile):
            logging.info("File %s already exists", outfile)
            return outfile

        logging.info("Collecting conversations and generating formatted file to %s...", outfile)
        character = character[0].upper() + character[1:] + ": "
        with open(outfile, "w", encoding="utf-8") as fo:
            for i in range(1, len(pages)):
                url = pages[i]

                response = urllib.request.urlopen(url, cafile=certifi.where())
                content = response.read()
                soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
                ps = soup.find_all('p', class_='MsoNormal') if i < 19 else \
                    soup.find_all('div', class_='entrytext')[0].contents
                last = "#"
                flg = False if i < 19 else True

                for p in ps:
                    if flg:
                        if p.name == "p":
                            line = str(p.string)
                        else:
                            continue
                    else:
                        line = p.get_text()

                    if not line or ":" not in line or "Scene: " in line:
                        last = "#"
                        continue
                    if last == "#":
                        last = line
                        if last[-1] == "\n":
                            last = last[:-1]
                    else:
                        cur = line
                        if cur[-1] == "\n":
                            cur = cur[:-1]
                        if cur.startswith(character):
                            left = last[last.find(":") + 1:]
                            right = cur[cur.find(":") + 1:]
                            res = left + "|" + right + "\n"
                            if res and ":" not in res and "|" in res:
                                fo.write(res)
                        last = cur
        return outfile




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
        logging.info("Extracting conversations from %s...", file)
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

        return conversations

    def getOutFile(self):
        if os.path.isfile(self.outfile):
            logging.info("File %s already exists", self.outfile)
        else:
            conversations = self.getConversations()
            logging.info("Generating formatted conversations file to %s...", self.outfile)
            with open(self.outfile, 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter="|")
                for conv in conversations:
                    writer.writerow(conv)
        return self.outfile

    def split_lines(self):
        lines = open(self.outfile, encoding='utf-8').read().strip().split('\n')
        return [l.split('|') for l in lines]




def get_vocab_and_pairs(input_file, name):
    logging.info("Reading lines of %s...", input_file)
    lines = open(input_file, encoding="utf-8").read().strip().split('\n')
    vocab = Vocab(name)
    pairs = []
    for line in lines:
        pair = line.split("|")
        temp = [normalizeString(s) for s in pair]
        if len(temp) == 2 and len(temp[0].split(" ")) < MAX_LENGTH and \
                len(temp[1].split(" ")) < MAX_LENGTH:
            pairs.append(temp)
            vocab.add_sentence(temp[0])
            vocab.add_sentence(temp[1])
    vocab.trim_words(MIN_WORDCOUNTS)

    keep = []
    for left, right in pairs:
        flag = True

        for word in left.split(" "):
            if word not in vocab.word2index:
                flag = False
                break

        if flag:
            for word in right.split(" "):
                if word not in vocab.word2index:
                    flag = False
                    break
            if flag:
                keep.append([left, right])

    return vocab, keep

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15
MIN_WORDCOUNTS = 3
#BATCH_SIZE = 10


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, name):
        self.name = name
        self.initialize()
        self.trimmed = False

    def initialize(self):
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

    def trim_words(self, min):
        if self.trimmed:
            return
        self.trimmed = True

        keep = []

        for key, value in self.word2count.items():
            if value >= min:
                keep.append(key)

        logging.info("Kept %d out of %d words for words appeared at least %d times",
                     len(keep), len(self.word2count), min)

        self.initialize()

        for item in keep:
            self._add_word(item)


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_index]


def getBatches(voc, pairs, batch_size, iters):
    return [getBatch(voc, pairs, batch_size) for _ in range(iters)]

def getBatch(voc, pairs, batch_size):
    # batches = []
    # for i in range(iters):
    pairs = [random.choice(pairs) for _ in range(batch_size)]
    pairs = sorted(pairs, key=lambda x : len(x[0].split(" ")), reverse=True)
    inp, out = zip(*pairs)
    inp_tensor = [indexesFromSentence(voc, sentence) for sentence in inp]
    inp_lengths = torch.tensor([len(i) for i in inp_tensor]).to(device)
    inp_tensor = torch.tensor(list(itertools.zip_longest(*inp_tensor, fillvalue=PAD_index)),
                              dtype=torch.long, device=device)
    out_tensor = [indexesFromSentence(voc, sentence) for sentence in out]
    max_out_length = max([len(i) for i in out_tensor])
    out_tensor = torch.tensor(list(itertools.zip_longest(*out_tensor, fillvalue=PAD_index)),
                              dtype=torch.long, device=device)
    mask = torch.zeros_like(out_tensor, dtype=torch.uint8, device=device)
    mask[out_tensor != PAD_index] = 1
    return inp_tensor, inp_lengths, out_tensor, mask, max_out_length
    #     batches.append((inp_tensor, inp_lengths, out_tensor, mask, max_out_length))
    # return batches


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        dropout = 0 if n_layers else dropout
        self.dropout = dropout
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,
                          self.n_layers, self.dropout, bidirectional=True)

    def forward(self, input_batch, input_lengths, hidden=None):
        embedded = self.embedding(input_batch)
        outputs = embedded
        packed = torch.nn.utils.rnn.pack_padded_sequence(outputs, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, outputs_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, attention_model, n_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if n_layers == 1:
            dropout = 0
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.attention = LuongAttention(hidden_size, attention_model.lower())

    def forward(self, input, hidden, encoder_outputs):

        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        attn_weights = self.attention(output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        output = output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((output, context), 1)
        concat_output = self.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return self.softmax(output), hidden


class GreedyDecoding(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedyDecoding, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length=MAX_LENGTH):

        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, dtype=torch.long, device=device) * SOS_index
        tokens = torch.zeros([0], dtype=torch.long, device=device)
        scores = torch.zeros([0], device=device)
        for i in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            tokens = torch.cat((tokens, decoder_input), dim=0)
            scores = torch.cat((scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return tokens, scores


class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(LuongAttention, self).__init__()
        if method not in ["dot", "concat", "general"]:
            raise ValueError("%s is not a valid attentional method" % method)
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size) if method == "concat" else \
            nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_v = nn.Parameter(torch.ones(self.hidden_size)) #FloatTensor(1, self.hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.method = method

    def forward(self, hidden, outputs):
        out = self.score(hidden, outputs).t()
        out = self.softmax(out).unsqueeze(1)
        return out

    def score(self, hidden, outputs):
        if self.method == "dot":
            return torch.sum(hidden * outputs, dim=2)
        elif self.method == "general":
            return torch.sum(hidden * self.attention(outputs), dim=2)
        elif self.method == "concat":
            temp = self.attention(torch.cat((hidden.expand(outputs.size(0), -1, -1), outputs), 2)).tanh()
            return torch.sum(self.attention_v * temp, dim=2)


def train(in_tensor, tgt_tensor, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, in_lengths, max_tgt_len, mask, teacher_forcing_ratio):
    encoder.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss, total_count = 0, 0
    print_loss = []
    encoder_outputs, encoder_hidden = encoder(in_tensor, in_lengths)

    batch_size = in_tensor.size(1)

    decoder_input = torch.tensor([[SOS_index for _ in range(batch_size)]], dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if teacher_forcing:
        for i in range(max_tgt_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoder_input = tgt_tensor[i].view(1, -1)
            ceLoss = criterion(decoder_output, tgt_tensor[i], mask[i])
            count = mask[i].sum().item()
            print_loss.append(ceLoss.item() * count)

            loss += ceLoss
            total_count += count
    else:
        for i in range(max_tgt_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi[i][0] for i in range(batch_size)]], dtype=torch.long, device=device)

            ceLoss = criterion(decoder_output, tgt_tensor[i], mask[i])
            count = mask[i].sum().item()
            print_loss.append(ceLoss * count)
            loss += ceLoss
            total_count += count

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 50)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 50)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return float(sum(print_loss)) / total_count


def evaluate(encoder, decoder, vocab, character):
    encoder.eval()
    decoder.eval()

    greedy = GreedyDecoding(encoder, decoder)
    logging.info("Start communicating with our bot! Type 'quit' to quit")
    input_seq = normalizeString(input('> '))
    while input_seq.strip().lower() != "quit":
        try:

            inp = [indexesFromSentence(vocab, input_seq)]
            inp_length = torch.tensor([len(i) for i in inp], device=device)
            inp = torch.tensor(inp, dtype=torch.long, device=device).transpose(0, 1)
            tokens, scores = greedy(inp, inp_length)
            words = [vocab.index2word[item.item()] for item in tokens]
            output = ""
            for word in words:
                if word != EOS_token and word != PAD_token:
                    output += word + " "
        except KeyError:
            output = "sorry , i don t understand ."

        print(character + ": " + output.strip())
        input_seq = normalizeString(input('> '))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_size", default=500, type=int,
                    help="hidden size of encoder/decoder, also word vector size")
    ap.add_argument("--attn_model", default="dot",
                    help="type of attentional model: dot, general or concat")
    ap.add_argument("--batch_size", default=64, type=int,
                    help="batch size for training")
    ap.add_argument("--n_iters", default=2000, type=int,
                    help="total number of conversations to train on")
    ap.add_argument("--n_layers", default=2, type=int,
                    help="number of layers in encoder/decoder")
    ap.add_argument("--corpus", default="cornellMovie",
                    help="the corpus to train on: cornellMovie, TO BE ADDED")
    # TODO: add characters
    ap.add_argument("--character", help="the character to mimic, avaliable: sheldon ....")
    ap.add_argument("--dropout", default=0.1, type=float,
                    help="the dropout rate for training")
    ap.add_argument("--load_checkpoint", nargs=1,
                    help="checkpoint file to start from")
    ap.add_argument("--lr", default=0.0001, type=float,
                    help="initial learning rate")
    ap.add_argument("--lr_de", default=5.0, type=float,
                    help="decoder learning rate")
    ap.add_argument("--tf_ratio", default=0.5, type=float,
                    help="teacher forcing ratio")
    ap.add_argument("--print_every", default=1, type=int,
                    help="print loss info every this many training examples")
    ap.add_argument("--checkpoint_every", default=1000, type=int,
                    help="write out checkpoint every this many training examples")

    args = ap.parse_args()

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint[0])
        # TODO: LOAD STATE

    vocab, pairs = None, None

    if args.character is None: #args.corpus == "cornellMovie":
        corpus = CornellMovieCorpus()
        file = corpus.getOutFile()
        character = "Baseline bot"

        vocab, pairs = get_vocab_and_pairs(file, args.corpus)

    else:
        character = args.character.strip().lower()
        if character not in ["sheldon", "leonard"]:
            raise ValueError(args.character.strip().lower() + " is not a valid character")
        # TODO: add corpus
        else:
            if character in ["sheldon", "leonard"]:    # TBBT characters
                corpus = TBBTCorpus()
                file = corpus.getOutFile(character)
                vocab, pairs = get_vocab_and_pairs(file, character)
                pass
            pass

        pass

    encoder = EncoderRNN(args.hidden_size, vocab.n_words, n_layers=args.n_layers,
                         dropout=args.dropout).to(device)
    decoder = AttnDecoderRNN(vocab.n_words, args.hidden_size, args.attn_model,
                             n_layers=args.n_layers, dropout=args.dropout).to(device)

    encoder.train()
    decoder.train()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr_de * args.lr)

    if args.load_checkpoint:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])
        encoder_optimizer.load_state_dict(state['encopt_state'])
        decoder_optimizer.load_state_dict(state['decopt_state'])

    start = time.time()
    print_loss_total = 0
    iters = state['iter'] if args.load_checkpoint else 0

    batch_size = args.batch_size

    training_batches = getBatches(vocab, pairs, batch_size, args.n_iters)

    logs = []
    logging.info("Start training...")
    while iters < args.n_iters:
        iters += 1

        training_batch = training_batches[iters - 1]

        in_tensor, in_lengths, tgt_tensor, mask, max_tgt_len = training_batch

        loss = train(in_tensor, tgt_tensor, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, in_lengths, max_tgt_len, mask, args.tf_ratio)

        print_loss_total += loss
        logs.append(loss)

        if iters % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iters,
                         iters / args.n_iters * 100,
                         print_loss_avg)


    plt.plot(logs)
    plt.show()
    plt.savefig('Loss.png')
    evaluate(encoder, decoder, vocab, character)


if __name__ == "__main__":
    main()

