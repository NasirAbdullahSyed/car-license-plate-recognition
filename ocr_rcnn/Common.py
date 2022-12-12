import torch
import numpy as np
import os, math, re
import math
from textdistance import levenshtein as lev


# This function creates a directory if it deosn't already exists 
def wrap_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split(samples, **kwargs):
    total = len(samples)
    indices = list(range(total))
    if kwargs['random']:
        np.random.shuffle(indices)
    percent = kwargs['split']
    # Split indices
    current = 0
    train_count = np.int(percent * total)
    train_indices = indices[current:current + train_count]
    current += train_count
    test_indices = indices[current:]
    train_subset, test_subset = [], []
    for i in train_indices:
        train_subset.append(samples[i])

    for i in test_indices:
        test_subset.append(samples[i])
    return train_subset, test_subset

def text_align(prWords, gtWords):
    row, col = len(prWords), len(gtWords)
    adjMat= np.zeros((row, col), dtype=float)
    for i in range(len(prWords)):
        for j in range(len(gtWords)):
            adjMat[i, j] = lev.normalized_distance(prWords[i], gtWords[j])
    pr_aligned=[]
    for i in range(len(prWords)):
        nn = list(map(lambda x:gtWords[x], np.argsort(adjMat[i, :])[:1])) 
        pr_aligned.append((prWords[i], nn[0]))
    return pr_aligned


class Eval:
    def _blanks(self, max_vals,  max_indices):
        def get_ind(indices):
            result = []
            for i in range(len(indices)):
                if indices[i] != 0:
                    result.append(i)
            return result
        non_blank = list(map(get_ind, max_indices))
        scores = []

        for i, sub_list in enumerate(non_blank):
            sub_val = []
            if sub_list:
                for item in sub_list:
                    sub_val.append(max_vals[i][item])
            score = np.exp(np.sum(sub_val))
            if math.isnan(score):
                score = 0.0
            scores.append(score)
        return scores


    def _clean(self, word):
        regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”„'))
        return regex.sub('', word)

    def char_accuracy(self, pair):
        words, truths = pair
        words, truths = ''.join(words), ''.join(truths)
        sum_edit_dists = lev.distance(words, truths)
        sum_gt_lengths = sum(map(len, truths))
        fraction = 0
        if sum_gt_lengths != 0:
            fraction = sum_edit_dists / sum_gt_lengths

        percent = fraction * 100
        if 100.0 - percent < 0:
            return 0.0
        else:
            return 100.0 - percent

    def word_accuracy(self, pair):
        correct = 0
        word, truth = pair
        if self._clean(word) == self._clean(truth):
            correct = 1
        return correct

    def format_target(self, target, target_sizes):
        target_ = []
        start = 0
        for size_ in target_sizes:
            target_.append(target[start:start + size_])
            start += size_
        return target_

    def word_accuracy_line(self, pairs):
        preds, truths = pairs
        word_pairs = text_align(preds.split(), truths.split())
        word_acc = np.mean((list(map(self.word_accuracy, word_pairs))))
        return word_acc

class TrainingMonitor:
    def __init__(self, save_file_path, stop_count=5, verbose=False, delta=0, best_score=None):
        self.counter = 0
        self.delta = delta
        self.stop_count = stop_count
        self.verbose = verbose
        self.best_score = best_score
        self.stop_training = False
        self.prev_val_loss = np.Inf
        self.save_file_path = save_file_path

    def __call__(self, val_loss, epoch, model, optimizer):
        score = -val_loss
        state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'best': score
                }
        if self.best_score is None:
            self.best_score = score
            self.save_model(val_loss, state)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print("Training Monitor Observed Increase in Validation Loss.")
            print(f'Model Training Stop counter:  [{self.counter}]/[{self.stop_count}]')
            if self.counter >= self.stop_count:
                self.stop_training = True
        else:
            self.best_score = score
            self.save_model(val_loss, state)
            self.counter = 0

    def save_model(self, val_loss, state):
        if self.verbose:
            print(f'Validation Loss Decreased [{self.prev_val_loss:.6f} --> {val_loss:.6f}]')
        torch.save(state, self.save_file_path)
        print(f"Model saved at '{self.save_file_path}'")
        self.prev_val_loss = val_loss


class AccuracyMeasure:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1 * float("inf")
        self.min = float("inf")

    def add(self, element):
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        if self.count == 0:
            return float("inf")
        return self.total / self.count

    def __str__(self):
        return "%s (min, avg, max): (%.3lf, %.3lf, %.3lf)" % (self.name, self.min, self.compute(), self.max)
        
class LabelCodec:
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {'': 0}
        for i, char in enumerate(alphabet):
            # 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        length = []
        result = []
        for item in text:
            length.append(len(item))
            for char in item:
                if char in self.dict:
                    index = self.dict[char]
                else:
                    index = 0
                result.append(index)

        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            if text.numel() != length:
                raise ValueError(f"Text Length ({text.numel()}) and Declared Length ({length}) Mismatch!")
            if raw:
                return ''.join([self.alphabet[i - 1] for i in text])
            else:
                char_list = []
                for i in range(length):
                    if text[i] != 0 and (not (i > 0 and text[i - 1] == text[i])):
                        char_list.append(self.alphabet[text[i] - 1])
                return ''.join(char_list)
        else:
            # Batch Mode Decoding
            if text.numel() != length.sum():
                raise ValueError(f"Text Length ({text.numel()}) and Declared Length ({length.sum()}) Mismatch!")
            texts = []
            index = 0
            for i in range(length.numel()):
                offset = length[i]
                texts.append(self.decode(text[index:index + offset], torch.IntTensor([offset]), raw=raw))
                index += offset
            return texts