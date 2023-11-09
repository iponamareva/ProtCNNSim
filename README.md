# ProtCNNSim: Discovery of new protein family relationships with Deep Learning

## Data
Per-family embeddings and their Pfam labels can be downloaded from Zenodo [10.5281/zenodo.10091909](https://zenodo.org/records/10091910).

Pfam 35 family-to-clan mapping can be downloaded from the [Pfam FTP site](https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.clans.tsv.gz) (direct download link). 

Embeddings, labels and Pfam-A clan descriptions should be located in the ```data/``` directory.

## Calculating scores

```ProtCNNSim_examples.ipynb``` contains the code for calculating the scores and building the sensitivity curve.

All utilities needed for this are in ```utils.py```.


