import networkx as nx
import random
# import sys
# sys.path.append('..')

import transformer.Constants as Constants

# 1.构造图
directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
edge_file = directory_path + "{dataset}_edge_file.txt".format(dataset=Constants.DATASET)

node_num = Constants.TYPE_NUMBER

dict_ = {}
print('Loading data ..')
user_traj = [[] for i in range(Constants.USER_NUMBER)]
train_file = '{dataset}_train.txt'.format(dataset=Constants.DATASET)
all_train_data = open(directory_path + train_file, 'r').readlines()
for eachline in all_train_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), int(lid), int(times)
    user_traj[uid].append([lid, times])
    dict_[lid] = uid

tune_file = '{dataset}_tune.txt'.format(dataset=Constants.DATASET)
all_tune_data = open(directory_path + tune_file, 'r').readlines()
for eachline in all_tune_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), int(lid), int(times)
    user_traj[uid].append([lid, times])
    dict_[lid] = uid


def createGraph():
    f = open(edge_file, "w")
    for traj in user_traj:
        for i in range(len(traj)-1):
            [lid, times] = traj[i]
            [lid2, times2] = traj[i+1]
            node_from, node_to = lid, lid2
            if node_from != node_to:
                f.write("%s %s %s\n" % (node_from, node_to, 1))
    f.close()


print('Creating Graph ..')
createGraph()


# 2.读图
def loadGraph(fileName):
    G = nx.read_edgelist(fileName, create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])
    return G


G = loadGraph(edge_file)


# # 新增一列
# f.write("%s %s %s %s\n" % (a, b, weight, weight2))
# # data增加对应列名称与属性
# data=[('weight', float), ('weight2', int)]

# # 3.获取图节点
# print(G.nodes())
# print(G.number_of_nodes())


# # 4.获取节点邻居
# for node in G.nodes:
#     nbrs = G.neighbors(node)
#     weights = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]

# # 添加节点
# G.add_node('10')
# # 批量添加
# G.add_nodes_from(['1', '2', '3'])
# # 单独删除
# G.remove_node('10')
# # 批量删除
# G.remove_nodes_from(['1', '2', '3'])
#
# # 添加边
# G.add_edge('1', '2')
# G.add_edges_from([('1', '2'), ('3', '4')])
# G.remove_edge('1', '2')
# G.remove_edges_from([('1', '2'), ('3', '4')])
#
# # 添加权重
# G.add_edge('1', '2')
# G.edges['1', '2']['weight'] = 0.33

# import matplotlib.pyplot as plt
#
# kwargs = {"node_size": 30, "node_color": 'r', "node_shape": 'o',
#               "alpha": 0.5, "width": 0.5, "edge_color": 'black',"style": "solid",
#               "with_labels": "true", "font_size": 12, "font_color": "black"}
# nx.draw(G, **kwargs)
# plt.title("NodeAndEdge", fontsize=20)
# plt.show()


# 2.在图中游走获取序列
def deep_walk(all_nodes, walk_length=100):
    walks = []
    all_nodes = list(all_nodes)
    random.shuffle(all_nodes)
    for node in all_nodes:
        walk = [node]
        while len(walk) < walk_length:
            cur_walk = walk[-1]
            cur_neighbor = list(G.neighbors(cur_walk))
            if len(cur_neighbor) > 0:
                walk.append(random.choice(cur_neighbor))
            else:
                break
        walks.append(walk)
    return walks


print('Deep Walking ..')
walks = deep_walk(G.nodes)
# walks1 = deep_walk(G.nodes)
# walks2 = deep_walk(G.nodes)
# walks3 = deep_walk(G.nodes)
# walks4 = deep_walk(G.nodes)
# walks.extend(walks1)
# walks.extend(walks2)
# walks.extend(walks3)
# walks.extend(walks4)
print("# of valid walks", len(walks))
# # take 5 samples
# print("-----------------------")
# random_seq = walks[0: 200]
# for i in random_seq:
#     print(i)
# print("-----------------------")


from gensim.models import Word2Vec
import transformer.Constants as Constants

# walks = 200

print('Word2Vecing ..')

kwargs = {"sentences": walks, "min_count": 0, "vector_size": 512,
              "sg": 1, "hs": 0, "workers": 3, "window": 5}
model = Word2Vec(**kwargs)
# sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
#                  max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
#                  sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
#                  trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
#                  max_final_vocab=None


def get_embeddings(w2v_model, graph):
    count = 0
    invalid_word = []
    _embeddings = {}
    for word in graph.nodes():
        if word in w2v_model.wv:
            _embeddings[word] = w2v_model.wv[word]
        else:
            invalid_word.append(word)
            count += 1
    print("# of invalid words", len(invalid_word))
    print("# of valid embedding", len(_embeddings))

    return _embeddings

import numpy as np
import time

embeddings = get_embeddings(model, G)

print("Saving ..")
poi_embedding = np.zeros([Constants.TYPE_NUMBER, 512])
for key in embeddings:
    poi_embedding[int(key)] = np.array(embeddings[key])
# Y = np.array(poi_embedding)
np.save('{}_poi_embedding.npy'.format(Constants.DATASET), poi_embedding)
# print(embeddings)
# print(embeddings['46'])


y_ = [1, 501, 1001, 1501, 2001, 2501, 3001, 3501, 4001, 4501, 5001, 5501, 6001, 6501, 7001, 7501, 8001, 8501, 9001,
      9501, 10001, 10501, 11001, 11501, 12001, 12501, 13001, 13501, 14001, 14501, 15001, 15501, 16001, 16501, 17001,
      17501, 18001, 18501]
version = time.time()
poi_indices = []
y = []
index_ = 0
for key in y_:
    for value in user_traj[key]:
        poi_indices.append(value)
        y.append(index_)
    index_ += 1
poi_indices = np.array(poi_indices)
poi_label = np.array(y)

X, Y = [], []
for key in poi_indices:
    X.append(embeddings[str(key[0])])
    # Y.append(poi_label)
X = np.array(X)
Y = np.array(poi_label)



# print(embeddings)
# print(embeddings['46'])

import tsne
tsne.visualization(X, Y, '{}'.format(time.time()))