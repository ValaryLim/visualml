import numpy as np
from sklearn import datasets

def generate_data(dataset, n_samples = 100, noise = 0):
    if dataset == 'moons':
        return datasets.make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=0
        )

    elif dataset == 'circles':
        return datasets.make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=1
        )

    elif dataset == 'linear':
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            'Data type incorrectly specified. Please choose an existing '
            'dataset.')