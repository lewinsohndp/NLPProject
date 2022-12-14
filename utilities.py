import torch

"""Script with helpful functions"""

def createDataTuples(sequences):
    """creates tuples to go into batching"""
    dataTuples = []
    c = 1
    for sp,info in sequences:
        protein = "protein"+str(c)
        dataTuples.append( (protein,info['sequence']) )
        c+=1
    return dataTuples

def mask_seq(seq):
    """return list with one of each AA in seq masked"""
    new_seqs = []
    for i in range(len(seq)):
        #if i % 20 != 0: continue
        new_seq = ""
        for j, char in enumerate(seq):
            
            if j == i:
                new_seq += "<mask>"
            else: new_seq += char
        
        new_seqs.append(new_seq)
    return new_seqs

def forward_pass(model, alphabet, seqs):
    """forward pass all the way to target layer"""

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    data = [("blah", x) for x in seqs]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    return results["logits"]

def forward_pass_embed(model, alphabet, seqs):
    """forward pass to embedding layer"""
    
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    data = [("blah", x) for x in seqs]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

    return sequence_representations