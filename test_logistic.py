import numpy as np
import logistic

def test_logistic_regression():
    pos, neg = logistic.generate_well_separable(10000, 0.50)

    #graph_pos_neg(pos, neg)

    X = np.vstack([pos, neg])
    y = np.hstack([np.array([1] * len(pos)), 
                   np.array([0] * len(neg)),])
    data = logistic.generate_random_points(100, 
                                           center=np.array([2,2]), 
                                           scale=np.array([5,5]))
    theta = logistic.logistic_gradient_ascent(X, y)
    labels = logistic.label_data(data, theta, binarize=True)
    assert len([l for l in labels if l == 0]) > 10
    assert len([l for l in labels if l == 1]) > 10

    small_data = np.array([[-1, -1], [11, 11]])
    labels2 = logistic.label_data(small_data, theta, binarize=True)
    assert np.allclose([0, 1], labels2)
    assert not np.allclose([1, 1], labels2)

    #TODO: jperla: test split_labeled_data()
    #graph_labeled_data(data, labels) 
