import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Create NPY matrices for the subset of 85k unlabeled tweets

model = SentenceTransformer("all-mpnet-base-v2")

id2pos = {}; id2text = {}

text = []; sbert = []
dataset = json.load(open('all_data/random_85k_unlabeled.json'))
for i, key in enumerate(dataset):
    text.append(dataset[key]['text'])
    id2text[key] = dataset[key]['text']
    if key not in id2pos:
        id2pos[key] = []
    id2pos[key].append(i)

embeddings = model.encode(text)
text = np.array(text)
embeddings = np.array(embeddings)

np.save('all_data/text.npy', text)
np.save('all_data/sbert.npy', embeddings)

with open('all_data/id2pos.json', 'w') as fp:
    json.dump(id2pos, fp)

with open('all_data/id2text.json', 'w') as fp:
    json.dump(id2text, fp)
