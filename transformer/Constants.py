TOP_N = 10
DATASET = "Foursquare"
PAD = 0

NEGATIVE_NUM = 10

user_dict = {
    'Yelp': 30887,
    'Gowalla': 5628,
    'Yelp2020': 28038,
    'Foursquare': 7642,
    'movielen': 6040,
}

poi_dict = {
    'Yelp': 18995,
    'Gowalla': 31803,
    'Yelp2020': 15745,
    'Foursquare': 28483,
    'movielen': 3952
}

# Yelp challenge 18995  Gowalla  31803 	Yelp-2020 15745
TYPE_NUMBER = poi_dict.get(DATASET)
# Yelp challenge 30887  Gowalla 5628 Yelp-2020 28038
USER_NUMBER = user_dict.get(DATASET)

