import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uniprot
import utilities
from tqdm import tqdm
import pickle

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
seqids, fastas = uniprot.read_fasta('/Users/daniel/desktop/cp341/uniprot_sprot.fasta')

species_seqids = []
for seqid in seqids:
    temp = [x for x, j in enumerate(fastas[seqid]['description'].split()) if j[0:3] == "OS="][0]
    #print(temp)
    try:
        #print()
        #print(temp)
        first_part = fastas[seqid]['description'].split()[temp]
        if fastas[seqid]['description'].split()[temp+1][0:3] != "OX=":
            first_part += " " + fastas[seqid]['description'].split()[temp+1]
        
        if first_part == "OS=Bacillus subtilis":
            species_seqids.append(seqid)
    except IndexError:
        print(fastas[seqid])
        print(temp)
        break
np.random.seed(8)
chosen = np.random.choice(range(len(species_seqids)), size=1000, replace=False)

# mask amino acid, take BCE loss of softmax(predicted) from actual, average for all amino acid of same type, compare across
aa_loss = {}
aa_counts = {}
for i in tqdm(chosen):
    seq = fastas[species_seqids[i]]["sequence"]
    if len(seq) > 200: continue
    seq_list = utilities.mask_seq(seq)
    #for temp_seq in seq_list:
    #get all logit for entire sequence
    results = None
    for x in tqdm(range(len(seq_list))):
        temp_result = utilities.forward_pass(model, alphabet, [seq_list[x]])
        if results == None:
            results = temp_result
        else:
            results = torch.concat((results, temp_result), dim=0)
        #print(results.shape)
        #if i == 10: break
    #print(results)
    # go through, subset to just maxed amino acid, softmax of predicted layer, BCE with real
    for j, result in enumerate(results):
        masked_aa = seq[j]
        correct_row = result[j+1]
        softmax = torch.nn.Softmax(dim=-1)
        soft_row = softmax(correct_row)
        bce = torch.nn.CrossEntropyLoss()
        target = torch.tensor([alphabet.tok_to_idx[masked_aa]])
        loss = bce(torch.reshape(soft_row, (1,33)), target)
        try:
            aa_loss[masked_aa] += float(loss)
        except KeyError:
            aa_loss[masked_aa] = float(loss)
        
        try:
            aa_counts[masked_aa] += 1
        except KeyError:
            aa_counts[masked_aa] = 1
    
    with open('loss.pickle', 'wb') as handle:
        pickle.dump(aa_loss, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open('counts.pickle', 'wb') as handle:
        pickle.dump(aa_counts, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        #print(i)
    #print(results)

