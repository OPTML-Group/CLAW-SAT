import pickle
import gzip
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--random_views_path', action='store', dest='random_views_path')
parser.add_argument('--adv_views_path', action='store', dest='adv_views_path')
parser.add_argument('--joined_views_path', action='store', dest='joined_views_path')
args = parser.parse_args()

random_views = pickle.load(gzip.open(args.random_views_path, 'rb'))
adv_views = pickle.load(gzip.open(args.adv_views_path, 'rb'))

n = len(random_views)
assert n == len(adv_views)

joined_views = []
for i in range(n):
    assert random_views[i][0] == adv_views[i][0]
    new_example = random_views[i][:] + [adv_views[i][1]]
    joined_views.append(new_example)


with gzip.open(args.joined_views_path, 'wb') as f:
    pickle.dump(joined_views, f)

print('Saved adv + random views.')


