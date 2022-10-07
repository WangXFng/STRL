import networkx as nx
import random
# import sys
# sys.path.append('..')

import transformer.Constants as Constants

from gensim.models import Word2Vec
import transformer.Constants as Constants
import numpy as np
import time

# walks = 200

directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
user_traj = [[] for i in range(Constants.USER_NUMBER)]
train_file = '{dataset}_train.txt'.format(dataset=Constants.DATASET)
all_train_data = open(directory_path + train_file, 'r').readlines()
for eachline in all_train_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), lid, times
    user_traj[uid].append(lid)

tune_file = '{dataset}_tune.txt'.format(dataset=Constants.DATASET)
all_tune_data = open(directory_path + tune_file, 'r').readlines()
for eachline in all_tune_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), lid, times
    user_traj[uid].append(lid)

# print(user_traj[0])
# print(user_traj[1])

print('Word2Vecing ..')

kwargs = {"sentences": user_traj, "min_count": 0, "size": 512,  # vector_
              "sg": 1, "hs": 0, "workers": 3, "window": 300}
model = Word2Vec(**kwargs)
# sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
#                  max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
#                  sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
#                  trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
#                  max_final_vocab=None

print("Saving ..")
# print(model.wv)
poi_embedding = np.zeros([Constants.TYPE_NUMBER, 512])
invalid_word = []
count = 0
# for key in model.wv.vocab:
#     print(key)
for i in range(Constants.TYPE_NUMBER):
    word = str(i)
    if word in model.wv:
        poi_embedding[i] = np.array(model.wv[word])
    else:
        invalid_word.append(word)
        count += 1

print("# of invalid words", len(invalid_word))
# print("# of valid embedding", len(_embeddings))
# Y = np.array(poi_embedding)
np.save('{}_poi_embedding.npy'.format(Constants.DATASET), poi_embedding)
# print(embeddings)
# print(embeddings['46'])


y_ = [
    1, 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001, 4501, 5001, 5501, 6001, 6501, 7001, 7501, 8001, 8501, 9001,
      # 9501, 10001, 10501, 11001, 11501, 12001, 12501, 13001, 13501, 14001, 14501, 15001, 15501, 16001, 16501, 17001,
      17501, 18001, 18501]
version = time.time()
poi_indices = []
is_existed = {}
y = []
index_ = 0
for key in y_:
    for value in user_traj[key]:
        if value not in is_existed:
            poi_indices.append(value)
            y.append(index_)
            is_existed[value] = 1
        else:
            index = poi_indices.index(value)
            del is_existed[value]
            del poi_indices[index]
            del y[index]
    index_ += 1
poi_indices = np.array(poi_indices)
poi_label = np.array(y)

X, Y = [], []
for key in poi_indices:
    X.append(poi_embedding[int(key)])
    # Y.append(poi_label)
X = np.array(X)
Y = np.array(poi_label)



# print(embeddings)
# print(embeddings['46'])

import tsne
tsne.visualization(X, Y, '{}'.format(time.time()))