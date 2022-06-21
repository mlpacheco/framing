import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import random

random.seed(42)
# Create NPY matrices for the subset of 85k unlabeled tweets

model = SentenceTransformer("all-mpnet-base-v2")

id2pos = {}; id2text = {}; id2explanations = {}

text = []; sbert = []
dataset = json.load(open('all_data/random_85k_unlabeled.json'))
for i, key in enumerate(dataset):
    text.append(dataset[key]['text'])
    id2text[key] = dataset[key]['text']
    if key not in id2pos:
        id2pos[key] = []
    id2pos[key].append(i)

all_keys = []; n_more = 0; n_total = 0
top_five = ['Political Factors and Implications', 'Policy Prescription and Evaluation', 'Crime and Punishment', 'Health and Safety', 'Security and Defense', 'Economic']
with open('all_data/predicted_data/predicted_frames.tsv') as fp:
    for pos, line in enumerate(fp):
        if pos == 0:
            keys = line.strip().split('\t')
        else:
            elems = line.strip().split('\t')[1:]
            if elems[0] not in id2pos:
                continue
            else:
                id2explanations[elems[0]] = {'role': 'None', 'immi_frame': 'None', 'policy_frame': [], 'narrative': 'None'}
                policy = [keys[i] for i, x in enumerate(elems) if x == "1" and keys[i] in top_five]
                immigration = [keys[i] for i, x in enumerate(elems) if x == "1" and ":" in keys[i]]
                role = [keys[i] for i, x in enumerate(elems) if x == "1" and keys[i] in ["Victim", "Threat", "Hero"]]
                narrative = [keys[i] for i, x in enumerate(elems) if x == "1" and keys[i] in ["Episodic", "Thematic"]]
                if len(role) >= 1:
                    # Chose one at random
                    curr_role = random.choice(role)
                    id2explanations[elems[0]]['role'] = curr_role
                if len(immigration) >= 1:
                    curr_immi = random.choice(immigration)
                    id2explanations[elems[0]]['immi_frame'] = curr_immi
                if len(policy) >= 1:
                    id2explanations[elems[0]]['policy_frame'] = policy
                if len(narrative) >= 1:
                    id2explanations[elems[0]]['narrative'] = narrative[0]

with open('all_data/id2explanations.json', 'w') as fp:
    json.dump(id2explanations, fp)

#print(Counter(all_keys))

embeddings = model.encode(text)
text = np.array(text)
embeddings = np.array(embeddings)

np.save('all_data/text.npy', text)
np.save('all_data/sbert.npy', embeddings)

with open('all_data/id2pos.json', 'w') as fp:
    json.dump(id2pos, fp)

with open('all_data/id2text.json', 'w') as fp:
    json.dump(id2text, fp)
