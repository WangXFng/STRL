from transformer import Constants
# from dataset import Foursquare
import numpy as np
import scipy
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix


def read_poi_coos():
    directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
    poi_file = '{dataset}_poi_coos.txt'.format(dataset=Constants.DATASET)
    poi_coos = {}
    poi_data = open(directory_path + poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)

    place_coords = []
    for k, v in poi_coos.items():
        place_coords.append([v[0], v[1]])

    return place_coords

def cal_place_pairwise_dist(place_coordinates):
    # this method calculates the pair-wise rbf distance
    gamma = 0.6
    place_correlation = rbf_kernel(place_coordinates, gamma=gamma)
    np.fill_diagonal(place_correlation, 0)
    place_correlation[place_correlation < 0.1] = 0
    place_correlation = csr_matrix(place_correlation)

    return place_correlation


def main():
    # try attention model
    # train_matrix, test_set, place_coords = Foursquare().generate_data()
    place_correlation = cal_place_pairwise_dist(read_poi_coos())
    scipy.sparse.save_npz('./data/{dataset}/place_correlation_gamma60.npz'.format(dataset=Constants.DATASET), place_correlation)


if __name__ == '__main__':
    main()
