import networkx as nx
import random
# import sys
# sys.path.append('..')

import transformer.Constants as Constants

# 1.构造图
directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
edge_file = "directory_path + {dataset}_edge_file.txt".format(dataset=Constants.DATASET)

node_num = Constants.TYPE_NUMBER

print('Loading data ..')
user_traj = [[] for i in range(Constants.USER_NUMBER)]
train_file = '{dataset}_train.txt'.format(dataset=Constants.DATASET)
all_train_data = open(directory_path + train_file, 'r').readlines()
for eachline in all_train_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), int(lid), int(times)
    user_traj[uid].append([lid, times])

tune_file = '{dataset}_tune.txt'.format(dataset=Constants.DATASET)
all_tune_data = open(directory_path + tune_file, 'r').readlines()
for eachline in all_tune_data:
    uid, lid, times = eachline.strip().split()
    uid, lid, times = int(uid), int(lid), int(times)
    user_traj[uid].append([lid, times])


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

# import matplotlib.pyplot as plt
#
# kwargs = {"node_size": 30, "node_color": 'r', "node_shape": 'o',
#               "alpha": 0.5, "width": 0.5, "edge_color": 'black',"style": "solid",
#               "with_labels": "true", "font_size": 12, "font_color": "black"}
# nx.draw(G, **kwargs)
# plt.title("NodeAndEdge", fontsize=20)
# plt.show()


# 2.在图中游走获取序列
def deep_walk(all_nodes, walk_length=120):
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
# take 5 samples
print("-----------------------")
random_seq = walks[0: 200]
for i in random_seq:
    print(i)
print("-----------------------")

