import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input', nargs=2)

df1, df2 = parser.parse_args().input

df1 = pd.read_pickle(df1)
df2 = pd.read_pickle(df2)

df1.sort_values(by=['file_name', 'frame_number'])
df2.sort_values(by=['file_name', 'frame_number'])

assert df1.shape == df2.shape

scores = []

for i in range(df1.shape[0]):
    assert df1['file_name'][i] == df2['file_name'][i]
    assert df1['frame_number'][i] == df2['frame_number'][i]
    scores.append(np.mean(np.abs(df1['value'][i] - df2['value'][i]).astype(float)))

print ("Average difference: %s" % np.mean(scores))
