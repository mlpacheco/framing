import json
import argparse
import datetime
from twarc.client2 import Twarc2
from twarc.expansions import ensure_flattened
from tqdm import tqdm
import os

def main(args):
    # Your bearer token here
    config = json.load(open(args.config))
    t = Twarc2(consumer_key=config['consumer_key'], consumer_secret=config['consumer_secret'])

    tweet_ids = []
    #with open('all_data/predicted_data/predicted_frames.tsv') as fp:
    for subdir in ['dev', 'test', 'train']:
        with open(os.path.join('all_data', 'annotated_data', subdir, 'all_frames.tsv')) as fp:
            for i, line in enumerate(fp):
                if i == 0:
                    keys = line.rstrip().split('\t')
                else:
                    values = line.split('\t')
                    twid = values[1]
                    tweet_ids.append(twid)
    print(len(tweet_ids))

    for i in range(0, len(tweet_ids), 50000):
        curr_tweet_ids = tweet_ids[i:i+50000]
        results = {}

        pbar = tqdm(total=len(curr_tweet_ids))
        res = t.tweet_lookup(curr_tweet_ids)
        for elem in res:
            if 'data' in elem:
                for hyd in elem['data']:
                    results[hyd['conversation_id']] = hyd
                    pbar.update(1)
            else:
                print(elem.keys())
        pbar.close()
        print('dumping {}...'.format(i))
        #with open('all_data/predicted_data/hydrated_{}.json'.format(i), 'w') as fp:
        with open('all_data/annotated_data/hydrated_{}.json'.format(i), 'w') as fp:
            json.dump(results, fp)
        print("DONE")

    '''
    results = {}
    for ann in annotations:
        print(ann['Corpus'])
        tweet_ids = []
        pbar = tqdm(total=len(ann['Tweets']))
        for tw in ann['Tweets']:
            tweet_id = tw['tweet_id']
            tweet_ids.append(tweet_id)
        res = t.tweet_lookup(tweet_ids)
        for elem in res:
            if 'data' in elem:
                for hyd in elem['data']:
                    results[hyd['conversation_id']] = hyd
                    pbar.update(1)
            else:
                print(elem.keys())
        pbar.close()

    with open("/scratch1/pachecog/covid/MFTC_V4_hydrated.json", "w") as fp:
        json.dump(results, fp)
    '''

    '''
    hydrated = json.load(open("/scratch1/pachecog/covid/MFTC_V4_hydrated.json"))

    for ann in annotations:
        for tw in ann['Tweets']:
            tweet_id = tw['tweet_id']
            if tweet_id in hydrated:
                tw['text'] = hydrated[tweet_id]['text']

    with open('/scratch1/pachecog/covid/MFTC_V4.json', "w") as fp:
        json.dump(annotations, fp)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    main(args)
