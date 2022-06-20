import json
import os
import random

# First check labeled data to get a sense of distribution
for file in os.listdir('all_data/annotated_data'):
    if file.endswith('.json'):
        with open('all_data/annotated_data/{}'.format(file)) as fp:
            tweets = json.load(fp)

print("annotated", len(tweets.keys()))
created_at = []
for tw in tweets:
    date = tweets[tw]['created_at']
    date = date.split('-')
    date = date[0] + "-" + date[1]
    created_at.append(date)

min_date = min(created_at)
max_date = max(created_at)
print("min", min_date, "max", max_date)


pred_tweets = {}
for file in os.listdir('all_data/predicted_data'):
    if file.endswith('.json'):
        with open('all_data/predicted_data/{}'.format(file)) as fp:
            tweets = json.load(fp)
            for tw in tweets:
                date = tweets[tw]['created_at']
                date = date.split('-')
                date = date[0] + "-" + date[1]
                if date < min_date or date > max_date:
                    continue
                else:
                    pred_tweets[tw] = tweets[tw]

random_tweets = {}
random.seed(42)
print("unlabeled", len(pred_tweets))
keys = list(pred_tweets.keys())
random.shuffle(keys)
keys = keys[:85000]

for key in keys:
    random_tweets[key] = pred_tweets[key]

with open('all_data/random_85k_unlabeled.json', 'w') as fp:
    json.dump(random_tweets, fp)


