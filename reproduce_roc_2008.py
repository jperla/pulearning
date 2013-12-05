import os
import logging
import functools

import scipy
import sklearn
import numpy as np

import sgdlr
from fastgridsearch import FastGridSearchCV

#C_VALUES = [0.01, 0.1]
#C_VALUES = [1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2, 1e5, 1e8, 1e11, 1e14]
#C_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
#C_VALUES = [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,]
C_VALUES = [2**-8, 2**-7, 2**-6, 2**-5,]

def read_swissprot_data():
    """Reads in swissprot dataset from 3 files in proteindata folder.
        Returns 3-tuple of numpy arrays.
    """
    folder = 'proteindata'
    npy_filenames = 'pos', 'neg', 'test_pos'
    return (np.load(os.path.join(folder, 'data.%s.swissprot.npy' % d)) for d in npy_filenames)

def create_labels(*stacks):
    """Accepts a variable number of of 2-tuples, where the first element is a 0 or 1 label,
        and the second element is the number of such labels to append.
       Returns a 1-d array with lots of 0's and 1's in sequence.
    """
    labels = [np.array([label] * number) for label, number in stacks]
    return np.hstack(labels)

def calculate_roc(true_labels, estimated_labels):
    """Accepts two 1-d arrays of the same size.
       Returns the false positive rate array, true positive rate array (for graphic ROC),
            and the area under the ROC curve.
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels, estimated_labels)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc

class ROCCurve():
    def __init__(self, name, color, roc_auc, fpr, tpr):
        self.name = name
        self.color = color
        self.roc_auc = roc_auc
        self.fpr = fpr
        self.tpr = tpr

def fit_and_score(classifier, X, y, test_set, test_labels):
    """Fits the classifier to teh data, then tests it and generates a ROC curve.
        name parameter is used for logging.
    """
    # fit
    classifier.fit(X, y)
     
    # if this is a grid search, get the best estimator
    c = classifier.best_estimator_ if hasattr(classifier, 'best_estimator_') else classifier

    # predict
    logging.debug('params: %s' % c.get_params())
    probabilities = c.predict_proba(test_set)[:,1]

    # calculate ROC
    return calculate_roc(test_labels, probabilities)

def fit_and_generate_roc_curve(name, color, classifier, X, y, test_set, test_labels):
    logging.info('starting %s...' % name)
    fpr, tpr, roc_auc = fit_and_score(classifier, X, y, test_set, test_labels)
    logging.info('AUC for %s: %f' % (name, roc_auc))
    return ROCCurve(name, color, roc_auc, fpr, tpr)

if __name__=='__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    pos, neg, unlabeled_pos = read_swissprot_data()
    # switch cases to use a smaller labeled dataset
    if 'switch_cases' in locals() and switch_cases:
        logging.warning('Switching positive and negative datasets!')
        unlabeled_pos, pos = pos, unlabeled_pos 

    truncate = lambda m: m[:int(m.shape[0] / 30),:]
    # Use less data so that we can move faster, comment this out to use full dataset
    #pos, neg, unlabeled_pos = truncate(pos), truncate(neg), truncate(unlabeled_pos)

    num_folds = 10
    kfold_pos = list(sklearn.cross_validation.KFold(pos.shape[0], n_folds=num_folds, shuffle=True, random_state=0))
    kfold_neg = list(sklearn.cross_validation.KFold(neg.shape[0], n_folds=num_folds, shuffle=True, random_state=0))
    kfold_unlabeled_pos = list(sklearn.cross_validation.KFold(unlabeled_pos.shape[0], n_folds=num_folds, shuffle=True, random_state=0))

    for i in range(1):
        pos_indices_train, pos_indices_test = kfold_pos[i]
        neg_indices_train, neg_indices_test = kfold_neg[i]
        unlabeled_pos_indices_train, unlabeled_pos_indices_test = kfold_unlabeled_pos[i]

        pos_train, pos_test = pos[pos_indices_train], pos[pos_indices_test]
        neg_train, neg_test = neg[neg_indices_train], neg[neg_indices_test]
        unlabeled_pos_train, unlabeled_pos_test = unlabeled_pos[unlabeled_pos_indices_train], unlabeled_pos[unlabeled_pos_indices_test]

        test_set = np.vstack([pos_test, unlabeled_pos_test, neg_test])
        test_labels = create_labels((1, pos_test.shape[0] + unlabeled_pos_test.shape[0]),
                                    (0, neg_test.shape[0]))

        calculate_test_roc = functools.partial(calculate_roc, test_labels)

        logging.debug('pos train: %s', str(pos_train.shape))
        logging.debug('neg train: %s', str(neg_train.shape))
        logging.debug('unlabeled pos train: %s', str(unlabeled_pos_train.shape))

        # set up the datasets
        X = np.vstack([pos_train, unlabeled_pos_train, neg_train])
        y = create_labels((1, pos_train.shape[0]),
                          (0, unlabeled_pos_train.shape[0] + neg_train.shape[0]))
        y_labeled = create_labels((1, pos_train.shape[0] + unlabeled_pos_train.shape[0]),
                                  (0, neg_train.shape[0]))
        X, y, y_labeled = sklearn.utils.shuffle(X, y, y_labeled)
        X = scipy.sparse.csr_matrix(X) # sparsify X

        lr_param_grid = {'alpha': [0.1, 0.01, 0.001], 'n_iter': [10, 30, 100, 200, 400, 1000]}

        roc_curves = []

        # sci-kit learn's sgd classifier
        name = 'Sci-Kit SGD LR L2-regularized pos-only labels'
        sgd = sklearn.grid_search.GridSearchCV(sklearn.linear_model.SGDClassifier(loss='log',
                                                                                  penalty='l2',
                                                                                  random_state=2),
                                               lr_param_grid, cv=3, n_jobs=-1)
        curve = fit_and_generate_roc_curve(name, 'p-', sgd, X, y, test_set, test_labels)
        roc_curves.append(curve)

        name = 'Sci-Kit SGD LR L2-regularized true labels'
        curve = fit_and_generate_roc_curve(name, 'p-', sgd, X, y_labeled, test_set, test_labels)
        roc_curves.append(curve)

        max_iter = 200
        name = 'Modified LR pos-only labels'
        mlr = sgdlr.SGDModifiedLogisticRegression(alpha=0.01, n_iter=max_iter)
        curve = fit_and_generate_roc_curve(name, 'r--', mlr, X, y, test_set, test_labels)
        roc_curves.append(curve)

        '''
        logging.info('b = %s' % mlr.b_)
        logging.info('1.0 / (1.0 + b*b) = %s' % (1.0 / (1.0 + mlr.b_**2)))
        '''

        lr = sklearn.grid_search.GridSearchCV(sgdlr.SGDLogisticRegression(),
                                              lr_param_grid, cv=3, n_jobs=-1)

        # Baseline if we knew everything
        name = 'LR true labels'
        curve = fit_and_generate_roc_curve(name, 'r-', lr, X, y_labeled, test_set, test_labels)
        roc_curves.append(curve)

        name = 'LR pos-only labels'
        curve = fit_and_generate_roc_curve(name, 'r-.', lr, X, y, test_set, test_labels)
        roc_curves.append(curve)

        calculate_svms = False
        if calculate_svms:
            param_grid = {'C': C_VALUES}
            svm = FastGridSearchCV(sklearn.svm.LinearSVC(),
                                   sklearn.svm.SVC(kernel='linear', probability=True, cache_size=2000),
                                   param_grid,
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=1)
            curve = fit_and_generate_roc_curve('SVM pos-only labels', 'b-.', svm, X, y, test_set, test_labels)
            roc_curves.append(curve)

            curve = fit_and_generate_roc_curve('SVM true labels', 'b-', svm, X, y_labeled, test_set, test_labels)
            roc_curves.append(curve)

            biased_svm_param_grid = {'class_weight': [{0: 1.0, 1: 1.0},
                                                      {0: 1.0, 1: 2.0},
                                                      {0: 1.0, 1: 10.0}]}
            biased_svm_param_grid.update(param_grid)
            biased_svm = FastGridSearchCV(sklearn.svm.LinearSVC(),
                                          sklearn.svm.SVC(kernel='linear', probability=True, cache_size=2000),
                                          biased_svm_param_grid,
                                          cv=3,
                                          n_jobs=-1,
                                          verbose=1)
            curve = fit_and_generate_roc_curve(name, 'g-', biased_svm, X, y_labeled, test_set, test_labels)
            roc_curves.append(curve)

            calculate_weighted_svm = False
            if calculate_weighted_svm:
                logging.info('starting weighted SVM...')
                # I have to copy to avoid an error that says that svm_weight is not C-contiguous!
                svm_weight = svm_label_data(X, y, X)[:,1].copy()
                # now run this again with the probabilites as the weights
                svm_labels = svm_label_data(X, y, test_set, sample_weight=svm_weight, C=[0.125, 0.25, 0.5, 1.0])
                fpr, tpr, roc_auc = calculate_test_roc(svm_labels[:,1])
                name = 'Weighted SVM pos-only labels'
                roc_curves.append((name, roc_auc, fpr, tpr, 'g--'))
                logging.info('AUC for %s: %f' % (name, roc_auc))

        # Plot ROC curve
        import pylab as pl
        pl.clf()

        sorted_roc_curves = list(reversed(sorted(roc_curves, key=lambda c: c.roc_auc)))
        for c in sorted_roc_curves:
            pl.plot(c.fpr, c.tpr, c.color, label='%s (AUC = %0.4f)' % (c.name, c.roc_auc))

        pl.plot([0, 1], [0, 1], 'k--')
        if sorted_roc_curves[-1].roc_auc < 0.8:
            # show the full curve if there are low roc_auc
            pl.xlim([0.0, 1.0])
            pl.ylim([0.0, 1.0])
        else:
            pl.xlim([0.0, 0.3])
            pl.ylim([0.7, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC for SwissProt')
        pl.legend(loc="lower right")
        pl.show()


