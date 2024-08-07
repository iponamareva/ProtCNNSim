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
  """
  Returns one-hot encoding for the amino acid sequence.
  
  Args:
      amino_acid_residues (str): Input sequence, containing characters from
      AMINO_ACID_VOCABULARY

  Returns:
     np.array of shape (len(amino_acid_residues), len(AMINO_ACID_VOCABULARY)).
  """
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
  """
  Returns mappings for family accessions and family indices in prediction tensors.

  Args:
      release (int): Pfam release number.

  Returns:
      dict, dict: Two dictionaries with mappings:
          - family index to family accession
          - family accession to family index
  """
  vocab_path = f'vocab_{release}.json'
  f = open(vocab_path, 'r')
  vocab = f.readline().strip('[]').split(', ')
  vocab = [x[1:-1] for x in vocab]
  r_vocab = dict((vocab[i], i) for i in range(len(vocab)))
  f.close()
  return vocab, r_vocab


def labels_to_preds(preds, vocab, TH=0.025, LTH=20, seq_len=None):
  """
  Converts per-residue predictions to domain calls.

  Args:
      preds (tf.Tensor): Tensor with model predictions.
      vocab (dict): Dictionary that maps family index to family accession (e.g. PF000001).
      TH (float): Probability threshold for per-residue predictions.
      LTH (int): Length threshold (minimal length) for domain calls.
      seq_len (int):

  Returns:
      defaultdict: A defaultdict where keys are family names (strings) and values are lists of tuples
        where tuples contain (int, int) values of the start and the end of the predicted domain.
  """ 
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


def get_signatures(saved_model):
  """
  Gets signatures for tensors needed for prediction.

  Args:
      saved_model (str): Path to the model.
      
  Returns:
      str, str, str, str: Names of the tensors.
  """
  output_signature = saved_model.signature_def['representation']
  repr_tensor_name = output_signature.outputs['output'].name
  output_signature_1 = saved_model.signature_def['label']
  labels_tensor_name = output_signature_1.outputs['output'].name
  sequence_input_tensor_name = saved_model.signature_def['representation'].inputs['sequence'].name
  sequence_lengths_input_tensor_name = saved_model.signature_def['representation'].inputs['sequence_length'].name
  
  return repr_tensor_name, labels_tensor_name, sequence_input_tensor_name, sequence_lengths_input_tensor_name


def run_model(path, sequences_df, release=35, pr_th=0.025, l_th=20):
  """
  Main function for running the model.
  
  Args:
      path (str): Path to the model.
      sequences_df (pandas.DataFrame): Dataframe with sequences data.
      release (int): Pfam database release.
      pr_th (float): Probability threshold for positive per-residue prediction.
      l_th (int): Length threshold for a single domain prediction.
      
  Returns:
      dict: dictionary with predictions and their annotations, with keys:
          - 'repr_list' (dict): A dictionary where keys are family accessions
             and values are averaged embeddings for each domain prediction.
          - 'repr_annot_all' (dict): A dictionary where keys are family accessions
             and values are lists of tuples (str, int, int): sequence accession,
             start and end of the domain calls.
  """
  repr_list = defaultdict(list)
  repr_annot_all = defaultdict(list)
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

        act_coords = labels_to_preds(labels, vocab, TH=pr_th, LTH=l_th)
        
        for fam in act_coords:
            for domain in act_coords[fam]:
                start, end = domain
                repr_list[fam].append(np.mean(embs[start:end], axis=0))
                repr_annot_all[fam].append((seq_acc, start, end))
            
  return {'repr_list': repr_list,
          'repr_annot_all': repr_annot_all
         }


def dump_results_split(results, VER, split_num, prefix='mean_embs'):
    """
    Saves the results from the model run to the specified path.

    Args:
        results (dict): A dictionary with run results for saving,
          returned by run_model function.
        VER (string): version of the experiment.
        split_num (int): Number of the sequences data split.
          Used when inference is done in splits on multiple nodes.
        prefix (str): Root directory name for saving.
    """
    results_path = f'{prefix}/VER_{VER}_split_{split_num}'
    os.mkdir(results_path)
    
    for result_key in results:
        with open(f'{results_path}/{result_key}.pkl', 'wb') as f:
            pickle.dump(results[result_key], f)
      
