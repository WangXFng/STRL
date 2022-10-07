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
poi_traj = [[] for i in range(Constants.TYPE_NUMBER)]
train_file = '{dataset}_train.txt'.format(dataset=Constants.DATASET)
all_train_data = open(directory_path + train_file, 'r').readlines()
for eachline in all_train_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), int(lid), times
    user_traj[uid].append(str(lid))
    poi_traj[lid].append(str(uid+Constants.TYPE_NUMBER))

tune_file = '{dataset}_tune.txt'.format(dataset=Constants.DATASET)
all_tune_data = open(directory_path + tune_file, 'r').readlines()
for eachline in all_tune_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), int(lid), times
    user_traj[uid].append(str(lid))
    poi_traj[lid].append(str(uid+Constants.TYPE_NUMBER))

for i, ut in enumerate(user_traj):
    ut.insert(int(len(ut)/2), str(i+Constants.TYPE_NUMBER))
for i, pt in enumerate(poi_traj):
    pt.insert(int(len(pt)/2), str(i))

# print(user_traj[0])
# print(user_traj[1])
user_traj.extend(poi_traj)

print('Word2Vecing ..')

d_model = 2048

kwargs = {"sentences": user_traj, "min_count": 0, "vector_size": d_model,  # vector_
              "sg": 1, "hs": 0, "workers": 3, "window": 300}
model = Word2Vec(**kwargs)
# sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
#                  max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
#                  sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
#                  trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
#                  max_final_vocab=None

print("Saving ..")
# print(model.wv)
poi_embedding = np.zeros([Constants.TYPE_NUMBER, d_model])
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

user_embedding = np.zeros([Constants.USER_NUMBER, d_model])
for i in range(Constants.USER_NUMBER):
    word = str(i+Constants.TYPE_NUMBER)
    if word in model.wv:
        user_embedding[i] = np.array(model.wv[word])
    else:
        invalid_word.append(word)

print("# of invalid words", len(invalid_word))
# print("# of valid embedding", len(_embeddings))
# Y = np.array(poi_embedding)
np.save('embeddings/{}_{}_poi_embedding.npy'.format(Constants.DATASET, d_model), poi_embedding)
np.save('embeddings/{}_{}_user_embedding.npy'.format(Constants.DATASET, d_model), user_embedding)
# print(embeddings)
# print(embeddings['46'])

if Constants.DATASET.__contains__('Gowalla'):
    y_ = [
        1, 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001, 4501, 5001, 5501]

elif Constants.DATASET.__contains__('Yelp2020'):
    y_ = [
        1, 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001, 4501, 5001, 5501, 6001, 6501, 7001, 7501,
        8001, 8501, 9001,
          9501, 10001, 10501, 11001, 11501, 12001, 12501, 13001, 13501, 14001, 14501, 15001]
    y_ = y_[1::2]
else:
    y_ = [
        1, 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001, 4501, 5001, 5501, 6001, 6501, 7001, 7501,
        8001, 8501, 9001,
          9501, 10001, 10501, 11001, 11501, 12001, 12501, 13001, 13501, 14001, 14501, 15001, 15501, 16001, 16501, 17001,
          17501, 18001, 18501]
    y_ = y_[1::2]
version = time.time()
poi_indices = []
user_indices = []
is_existed = {}
poi_label = []
user_label = []
index_ = 0
for key in y_:
    for i, value in enumerate(user_traj[key]):
        # continue
        # if i == int((len(user_traj[key])-1)/2):
        if int(value) >= Constants.TYPE_NUMBER:
            user_indices.append(int(value))
            user_label.append(index_)
        else:
            if value not in is_existed:
                # if int(value) < Constants.TYPE_NUMBER:
                poi_indices.append(int(value))
                poi_label.append(index_)
                is_existed[int(value)] = 1
            else:
                index = poi_indices.index(int(value))
                del is_existed[int(value)]
                del poi_indices[index]
                del poi_label[index]
    index_ += 1

# print(len(user_indices), len(y_))
poi_indices = np.array(poi_indices)
user_indices = np.array(user_indices)
# poi_label = np.array(poi_label)
# user_label = np.array(user_label)

X, Y = [], []
X2, Y2 = [], []
for key in poi_indices:
    X.append(poi_embedding[int(key)])
len_ = len(X)
for key in user_indices:
    # print(int(key))
    # print(int(key)-Constants.TYPE_NUMBER)
    X2.append(user_embedding[int(key)-Constants.TYPE_NUMBER])
X.extend(X2)
poi_label.extend(user_label)
X = np.array(X)
Y = np.array(poi_label)
# X2 = np.array(X)
# Y2 = np.array(user_label)

# print(embeddings)
# print(embeddings['46'])

import tsne
tsne.visualization(X, Y, '{}'.format(time.time()), len_, True)