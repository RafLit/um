import pandas as pd
import random
from sklearn.model_selection import train_test_split
def getData():
    attributeNames = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring','stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    print(len(attributeNames))
    data = pd.read_csv('agaricus-lepiota.data', names=attributeNames, index_col=False)
    data = data.iloc[:,:]
    lab = data.pop('class')
    return train_test_split(data, lab, test_size=0.3, random_state=1, shuffle=True)

