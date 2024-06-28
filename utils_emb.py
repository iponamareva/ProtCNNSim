import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow.compat.v1 as tf


_PFAM_GAP_CHARACTER = '-'
AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]

def residues_to_one_hot(amino_acid_residues):
  to_return = []
  normalized_residues = amino_acid_residues.replace('U', 'C').replace('O', 'X')
  for char in normalized_residues:
    if char in AMINO_ACID_VOCABULARY:
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index(char)] = 1.
      to_return.append(to_append)
    elif char == 'B':  # Asparagine or aspartic acid.
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index('D')] = .5
      to_append[AMINO_ACID_VOCABULARY.index('N')] = .5
      to_return.append(to_append)
    elif char == 'Z':  # Glutamine or glutamic acid.
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index('E')] = .5
      to_append[AMINO_ACID_VOCABULARY.index('Q')] = .5
      to_return.append(to_append)
    elif char == 'X':
      to_return.append(
          np.full(len(AMINO_ACID_VOCABULARY), 1. / len(AMINO_ACID_VOCABULARY)))
    elif char == _PFAM_GAP_CHARACTER:
      to_return.append(np.zeros(len(AMINO_ACID_VOCABULARY)))
    else:
      raise ValueError('Could not one-hot code character {}'.format(char))
  return np.array(to_return)
  

def get_vocabs(release=32):
  vocab_path = f'vocab_{release}.json'
  f = open(vocab_path, 'r')
  vocab = f.readline().strip('[]').split(', ')
  vocab = [x[1:-1] for x in vocab]
  r_vocab = dict((vocab[i], i) for i in range(len(vocab)))
  f.close()
  return vocab, r_vocab


def labels_to_preds(preds, vocab, TH=0.025, LTH=20, seq_len=None):
  if seq_len is not None:
      preds = preds[:seq_len, :]
    
  sum_acts = np.sum(preds, axis=0)
  act_coords = defaultdict(list)
  
  for famidx in np.asarray(sum_acts > TH * LTH).nonzero()[0]:
    start, end = -1, -1
    for i in range(preds.shape[0]):
      if preds[i][famidx] > TH:
        if start == -1:
          start = i
        end = i
      else:
        if end - start > LTH:
          act_coords[vocab[famidx]].append((start, end))
        start, end = -1, -1

  return act_coords
  

def make_storage():
    full_mean_embs = []
    full_embs = dict()
    repr_list = defaultdict(list)
    repr_annot_all = defaultdict(list)
    repr_annot_chosen = defaultdict(list)
    perseq_preds = dict()
    seq_accs = []
    return full_mean_embs, full_embs, repr_list, repr_annot_all, repr_annot_chosen, perseq_preds, seq_accs


def get_signatures(saved_model):
  output_signature = saved_model.signature_def['representation']
  repr_tensor_name = output_signature.outputs['output'].name
  output_signature_1 = saved_model.signature_def['label']
  labels_tensor_name = output_signature_1.outputs['output'].name
  sequence_input_tensor_name = saved_model.signature_def['representation'].inputs['sequence'].name
  sequence_lengths_input_tensor_name = saved_model.signature_def['representation'].inputs['sequence_length'].name
  
  return repr_tensor_name, labels_tensor_name, sequence_input_tensor_name, sequence_lengths_input_tensor_name


def run_model(path, sequences_df, release=35, pr_th=0.025):
  full_mean_embs, full_embs, repr_list, repr_annot_all, repr_annot_chosen, perseq_preds, seq_accs = make_storage()
  seq_annot = dict()
  
  vocab, r_vocab = get_vocabs(release)
  
  with tf.Graph().as_default():
    with tf.Session() as sess:
      saved_model = tf.saved_model.load(sess, ['serve'], path)
      repr_tensor_name, labels_tensor_name, sequence_input_tensor_name, sequence_lengths_input_tensor_name = get_signatures(saved_model)

      for i, row in sequences_df.iterrows():
        seq, seq_acc = row['sequence'], row['pfamseq_acc']
        
        logits = sess.run(
            [repr_tensor_name,
             labels_tensor_name],
            {
                sequence_input_tensor_name: [residues_to_one_hot(seq)],
                sequence_lengths_input_tensor_name: [len(seq)],
            }
        )
        embs, labels = logits
        embs, labels = embs[0], labels[0]

        act_coords = labels_to_preds(labels, vocab, TH=pr_th, LTH=20)
        perseq_preds[seq_acc] = act_coords
        
        for fam in act_coords:
            for domain in act_coords[fam]:
                start, end = domain
                repr_list[fam].append(np.mean(embs[start:end], axis=0))
                repr_annot_all[fam].append((seq_acc, start, end))
            
  return {'repr_list': repr_list,
          'repr_annot_all': repr_annot_all,
          'perseq_preds': perseq_preds
         }
