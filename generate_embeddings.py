import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uniprot
import utilities
from tqdm import tqdm

seqids, fastas = uniprot.read_fasta('/Users/daniel/desktop/cp341/uniprot_sprot.fasta')
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
# get human proteins

human_seqids = []
for seqid in seqids:
    temp = [x for x, j in enumerate(fastas[seqid]['description'].split()) if j[0:3] == "OS="][0]
    #print(temp)
    try:
        #print()
        #print(temp)
        first_part = fastas[seqid]['description'].split()[temp]
        if fastas[seqid]['description'].split()[temp+1][0:3] != "OX=":
            first_part += " " + fastas[seqid]['description'].split()[temp+1]
        
        if first_part == "OS=Homo sapiens":
            human_seqids.append(seqid)
    except IndexError:
        print(fastas[seqid])
        print(temp)
        break
np.random.seed(8)
chosen = np.random.choice(range(len(human_seqids)), size=1000, replace=False)

embeddings = []
index_id = []
for i in tqdm(chosen):
    seq = fastas[human_seqids[i]]["sequence"]
    if len(seq) > 1000: continue
    index_id.append(human_seqids[i])
    rep = utilities.forward_pass_embed(model, alphabet, [seq])
    embeddings.append(rep[0].numpy())

embeddings_df = pd.DataFrame(np.array(embeddings))
embeddings_df.index = index_id

embeddings_df.to_csv("human_protein_embeddings.csv")