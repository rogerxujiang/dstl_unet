import pandas as pd
import os
from shapely import wkt
import sys

data_dir = os.path.join('.', 'submission')

if __name__ == '__main__':

    files = []

    for cl in range(10):
        files.append(pd.read_csv(os.path.join(data_dir, 'class_{}.csv'.format(cl))))

    df = pd.read_csv(os.path.join(data_dir, '..', 'data/sample_submission.csv'))
    length = (len(df) - 1) / 100

    for idx, row in df.iterrows():

        shape = wkt.loads(files[row[1] - 1].iloc[idx, 2])

        if shape.is_valid:
            df.iloc[idx, 2] = files[row[1] - 1].iloc[idx, 2]
        else:
            print 'Index {}, ImageId {}, Class {} is not valid; fixing it;\n'.format(idx, row[0], row[1])
            shape1 = shape.buffer(0)
            assert shape1.is_valid
            df.iloc[idx, 2] = wkt.dumps(shape1)

        if not idx % length:
            sys.stdout.write('\r[' + '#' * (idx / length) + '-' * (100 - idx / length) + ']' + 'Working on image No. {}'.format(idx))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    df.to_csv(os.path.join(data_dir, 'valid_submission.csv'), index=False)
