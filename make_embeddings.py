import sys
import pandas as pd
import argparse
import tensorflow.compat.v1 as tf

from utils_emb import run_model, dump_results_split

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-v", "--version", type=int
    parser.add_argument("-d", "--data-file", type=str)
    parser.add_argument("-i", "--split_num")
    args = parser.parse_args(argv)
    return args.path, args.version, args.data_file, args.split_num
  
  
if __name__ == "__main__":
    path, version, sample_filename, split_num = main(sys.argv[1:])
    
    print('Arguments for run:', path, version, sample_filename)
  
    sequences_df = pd.read_csv(sample_filename, sep='\t')
    print(f"Num sequences loaded: {len(sequences_df)}")

    results = run_model(path, sequences_df)
    
    dump_results_split(results, verbose=True, VER=version, split_num=split_num, prefix='single_model_all_pfamseq/run_dumps')
    print(f'{sample_filename} processed successfully!')
