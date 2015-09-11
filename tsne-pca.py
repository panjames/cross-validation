import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

X = np.load("train_set")

tsne = TSNE(n_components=2, init='random', random_state=0)

Y = tsne.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)

plt.show()