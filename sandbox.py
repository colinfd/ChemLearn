import pickle
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import seaborn as sns

key = ['comp', 'bulk', 'facet', 'site', 'ads_a', 'ads_b', 'gas', 'dE']

df = pickle.load(open('pairs.pkl'))
