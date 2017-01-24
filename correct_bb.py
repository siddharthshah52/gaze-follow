import pandas as pd
import pdb

# provided = pd.read_csv('../gazefollow_data/train_annotations.txt', header=None, usecols=[0, 8, 9])
provided = pd.read_csv('../gazefollow_data/test_annotations.txt', header=None, usecols=[0, 6, 7, 8, 9])
provided = provided.drop_duplicates(subset=0)

# found = pd.read_csv('annotationsTrain.csv', header=None)
found = pd.read_csv('annotationTest.csv', header=None)
found[10] = found[1] + found[3]
found[11] = found[2] + found[4]

joined = pd.merge(provided, found, how='inner', left_on=0, right_on=0)

# print joined.shape

annotations = joined[joined.apply(lambda x: x[6] >= x[1] and x[6] <= x[10] and x[7] >= x[2] and x[7] <= x[11], axis=1)]
# print annotations.head(10)
# print annotations.shape
annotations = annotations.drop([10, 11], axis=1)
cols = range(5) + range(6, 10)
annotations = annotations[cols]
annotations.to_csv('annotations_final_test.csv', header=False, index=False)
pdb.set_trace()