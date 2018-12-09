import numpy as np
from sklearn.model_selection import train_test_split

inds = np.arange(1999)
train, test_and_val = train_test_split(inds, test_size=0.3)
val, test = train_test_split(test_and_val, test_size=0.5)

np.save('train.npy', train)
np.save('val.npy', val)
np.save('test.npy', test)

