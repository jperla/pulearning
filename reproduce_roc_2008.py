import os
import logging
import functools

import scipy
import sklearn
from sklearn.grid_search import GridSearchCV
import numpy as np

import lr
from fastgridsearch import FastGridSearchCV

#C_VALUES = [0.01, 0.1]
#C_VALUES = [1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2, 1e5, 1e8, 1e11, 1e14]
#C_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
#C_VALUES = [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,]
#C_VALUES = [2**-8, 2**-7, 2**-6, 2**-5,]
C_VALUES = [2**-8, 2**-7, 2**-6,]

speed_multiple = 1
if 'FULL_GRAPH' in locals() and FULL_GRAPH:
    USE_L2_REGULARIZED_LR = True
    USE_SVMS = True
else:
    speed_multiple = 10
    USE_L2_REGULARIZED_LR = False
    USE_SVMS = False

USE_SGD_SVM = True
USE_WEIGHTED_SVM = (USE_SVMS and True)

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

def fit_and_score(classifier, X, y, test_set, test_labels, sample_weight=None):
    """Fits the classifier to teh data, then tests it and generates a ROC curve.
        name parameter is used for logging.
    """
    # fit
    if sample_weight is not None:
        classifier.fit(X, y, sample_weight=sample_weight)
    else:
        # classifier.fit may not accept sample weight
        classifier.fit(X, y)
     
    # if this is a grid search, get the best estimator
    c = classifier.best_estimator_ if hasattr(classifier, 'best_estimator_') else classifier

    # predict
    logging.debug('params: %s' % c.get_params())
    try:
        probabilities = c.predict_proba(test_set)[:,1]
    except NotImplementedError:
        # TODO: This is not Platt scaling, dumb scaling
        scalars = c.decision_function(test_set)
        scalars -= np.min(scalars)
        probabilities = scalars / np.max(scalars)

    # calculate ROC
    fpr, tpr, roc_auc = calculate_roc(test_labels, probabilities)
    return c, fpr, tpr, roc_auc

def double_weight(X, y, probabilities, c):
    assert(probabilities.shape == y.shape)
    positive_indices, unlabeled_indices = (y == 1).nonzero()[0], (y == 0).nonzero()[0]
    positive, unlabeled = X[positive_indices], X[unlabeled_indices]
    upr = probabilities[unlabeled_indices]
    assert len(upr) == unlabeled.shape[0]

    unlabeled_probabilities = ((1.0 - c) / c) * (upr / (1.0 - upr))

    X2 = scipy.sparse.vstack([positive, unlabeled, unlabeled])
    y2 = np.hstack([np.array([1.0] * positive.shape[0]),
                    np.array([1.0] * len(unlabeled_probabilities)),
                    np.array([0.0] * len(unlabeled_probabilities))])
    sample_weight = np.concatenate([np.array([1.0] * positive.shape[0]),
                                    unlabeled_probabilities,
                                    1.0 - unlabeled_probabilities], axis=1)
    X2, y2, sample_weight = sklearn.utils.shuffle(X2, y2, sample_weight)
    return X2, y2, sample_weight

def fit_double_weighted(name, color, X, y, probabilities, c, test_set, test_labels,):
    X2, y2, sample_weight = double_weight(X, y, probabilities, c)

    # Learn on an SGD svm learner.
    wsvm = sklearn.linear_model.SGDClassifier(loss='hinge',
                                                penalty='l2',
                                                n_iter=200,
                                                alpha=0.01,
                                                random_state=0)
    # logistic regression instead of svm
    wsvm = sklearn.linear_model.SGDClassifier(loss='log',
                                                penalty='l2',
                                                n_iter=200,
                                                alpha=0.01,
                                                random_state=0)
    best_wsvm, curve = fit_and_generate_roc_curve(name, color, 
                                                  wsvm, X2, y2, test_set, test_labels, 
                                                  sample_weight=sample_weight)
    return best_wsvm, curve

def fit_and_generate_roc_curve(name, color, classifier, X, y, test_set, test_labels, sample_weight=None):
    logging.info('starting %s...' % name)
    c, fpr, tpr, roc_auc = fit_and_score(classifier, X, y, test_set, test_labels, sample_weight=sample_weight)
    logging.info('AUC for %s: %f' % (name, roc_auc))
    return c, ROCCurve(name, color, roc_auc, fpr, tpr)

if __name__=='__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    pos, neg, unlabeled_pos = read_swissprot_data()
    # switch cases to use a smaller labeled dataset
    cases_switched = False
    if 'SWITCH_CASES' in locals() and SWITCH_CASES:
        cases_switched = True
        logging.warning('Switching positive and negative datasets!')
        unlabeled_pos, pos = pos, unlabeled_pos 

    true_c = float(pos.shape[0]) / (pos.shape[0] + unlabeled_pos.shape[0])

    truncate = lambda m: m[:int(m.shape[0] / speed_multiple),:]
    # Use less data so that we can move faster, comment this out to use full dataset
    if speed_multiple > 1:
        pos, neg, unlabeled_pos = truncate(pos), truncate(neg), truncate(unlabeled_pos)

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
        N_FEATURES = 25000 # shrink number of features to test over-fitting
        X, test_set = X[:, :N_FEATURES], test_set[:, :N_FEATURES]
        # sparsify X
        X = scipy.sparse.csr_matrix(X)

        # scale
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        scaler.fit(X)
        X = scaler.transform(X)
        test_set = scaler.transform(test_set)

        roc_curves = []

        # POSONLY
        name = 'POLR pos-only labels'
        posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=200, eta0=0.1, c=None)
        best, curve = fit_and_generate_roc_curve(name, 'r-', posonly, X, y, test_set, test_labels)
        print 'b:', best.b_
        print 'c:', best.final_c()
        roc_curves.append(curve)

        if USE_L2_REGULARIZED_LR:
            # alpha here is a regularization constant
            sgd_param_grid = {'alpha': [0.001, 0.0001,],}

            # sci-kit learn's sgd classifier
            name = 'L2-regularized LR pos-only labels'
            sgd = sklearn.grid_search.GridSearchCV(sklearn.linear_model.SGDClassifier(loss='log',
                                                                                    n_iter=200,
                                                                                    random_state=2),
                                                sgd_param_grid, cv=3, n_jobs=-1)
            _, curve = fit_and_generate_roc_curve(name, 'p-', sgd, X, y, test_set, test_labels)
            roc_curves.append(curve)

            name = 'L2-regularized LR true labels'
            _, curve = fit_and_generate_roc_curve(name, 'p-', sgd, X, y_labeled, test_set, test_labels)
            roc_curves.append(curve)

        lr_param_grid = {'eta0': [0.01, 0.001,], 'n_iter':[200,],}

        major_case_b = 0.22941573387056188
        minor_case_b = 4.358898943540673
        mlr_param_grid = {} #'b': [major_case_b, 1.0, 2.0, 3.0, 4.0, 5.0]}
        mlr_param_grid.update(lr_param_grid)
        name = 'Ceiling LR pos-only labels'
        mlr = sklearn.grid_search.GridSearchCV(lr.SGDModifiedLogisticRegression(),
                                               mlr_param_grid, cv=3, n_jobs=-1)

        mlr.fit(X, y)
        logging.info('fit probabilities...')
        best_mlr = mlr.best_estimator_

        b = best_mlr.b_
        logging.info('b = %s' % b)
        logging.info('1.0 / (1.0 + b*b) = c = %s' % (1.0 / (1.0 + b**2)))

        # Get accuracy
        train_predicted = best_mlr.predict(X)
        logging.debug('\n' + sklearn.metrics.classification_report(y, train_predicted))
        #logging.debug('Log-likelihood of training set: %.4f' % best_mlr.log_likelihood(X, y, best_mlr.theta_, best_mlr.b_))

        probabilities = best_mlr.predict_proba(X)[:,1]
        c = (1.0 / (1.0 + b*b))
        _, curve = fit_double_weighted(name, 'r--', X, y, 
                                       probabilities, c,
                                       test_set, test_labels)
        roc_curves.append(curve)


        lr = sklearn.grid_search.GridSearchCV(lr.SGDLogisticRegression(),
                                              lr_param_grid, cv=3, n_jobs=-1)

        # Baseline if we knew everything
        name = 'LR true labels'
        _, curve = fit_and_generate_roc_curve(name, 'r-', lr, X, y_labeled, test_set, test_labels)
        roc_curves.append(curve)

        name = 'LR pos-only labels'
        _, curve = fit_and_generate_roc_curve(name, 'r-.', lr, X, y, test_set, test_labels)
        roc_curves.append(curve)

        if USE_SVMS:
            svm_param_grid = {'C': C_VALUES}
            sgd_svm_param_grid = {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001]}
            if USE_SGD_SVM:
                svm = GridSearchCV(sklearn.linear_model.SGDClassifier(loss='hinge', 
                                                                    penalty='l2', 
                                                                    n_iter=200, 
                                                                    random_state=0),
                                   sgd_svm_param_grid,
                                   cv=3,
                                   n_jobs=-1)
            else:
                svm = FastGridSearchCV(sklearn.svm.LinearSVC(),
                                    sklearn.svm.SVC(kernel='linear', probability=True, cache_size=2000),
                                    svm_param_grid,
                                    cv=3,
                                    n_jobs=-1,
                                    verbose=1)
            _, curve = fit_and_generate_roc_curve('SVM pos-only labels', 'b-.', svm, X, y, test_set, test_labels)
            roc_curves.append(curve)

            _, curve = fit_and_generate_roc_curve('SVM true labels', 'b-', svm, X, y_labeled, test_set, test_labels)
            roc_curves.append(curve)

            biased_svm_param_grid = {'class_weight': [{0: 1.0, 1: 1.0},] + [{0: 1.0, 1: 2.0},] +[{0: 1.0, 1: (i * 10.0)} for i in range(1, 21)],}
            biased_svm_param_grid = {'class_weight': [{0: 1.0, 1: 2.0},] +[{0: 1.0, 1: (i * 10.0)} for i in range(1, 3)],}
            if USE_SGD_SVM:
                biased_svm_param_grid.update(sgd_svm_param_grid)
                biased_svm = GridSearchCV(sklearn.linear_model.SGDClassifier(loss='hinge',
                                                                             penalty='l2',
                                                                             n_iter=200,
                                                                             random_state=0),
                                          biased_svm_param_grid,
                                          cv=3,
                                          n_jobs=-1)
            else:
                biased_svm_param_grid.update(svm_param_grid)
                biased_svm = FastGridSearchCV(sklearn.svm.LinearSVC(),
                                            sklearn.svm.SVC(kernel='linear', probability=True, cache_size=2000),
                                            biased_svm_param_grid,
                                            cv=3,
                                            n_jobs=-1,
                                            verbose=1)
            name = 'Biased SVM pos-only labels'
            _, curve = fit_and_generate_roc_curve(name, 'g-', biased_svm, X, y, test_set, test_labels)
            roc_curves.append(curve)

            if USE_WEIGHTED_SVM:
                logging.info('starting weighted SVM...')
                svm = GridSearchCV(sklearn.svm.SVC(kernel='linear', probability=True, cache_size=2000),
                                   svm_param_grid, cv=3, n_jobs=-1, verbose=3)
                svm.fit(X, y)
                logging.info('fit probabilities...')
                probabilities = svm.best_estimator_.predict_proba(X)[:,1]
                
                name = 'Weighted SVM pos-only labels'
                c = 0.5
                _, curve = fit_double_weighted(name, 'g--', X, y, probabilities, c, test_set, test_labels)
                roc_curves.append(curve)

        # Plot ROC curve
        import pylab as pl
        fig = pl.figure()

        sorted_roc_curves = list(reversed(sorted(roc_curves, key=lambda c: c.roc_auc)))
        for c in sorted_roc_curves:
            pl.plot(c.fpr, c.tpr, c.color, label='%s (AUC = %0.4f)' % (c.name, c.roc_auc))

        pl.plot([0, 1], [0, 1], 'k--')
        if sorted_roc_curves[-1].roc_auc < 0.8:
            # show the full curve if there are low roc_auc
            pl.xlim([0.0, 1.0])
            pl.ylim([0.0, 1.0])
        else:
            pl.xlim([0.0, 0.2])
            pl.ylim([0.8, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        title = 'ROC for Inverted SwissProt' if 'SWITCH_CASES' in locals() and not SWITCH_CASES else 'ROC for SwissProt'
        pl.title(title)
        pl.legend(loc="lower right")

        name = 'rocswappedproteindata' if cases_switched else 'rocproteindata'
        fig.savefig('pdf/%s.png' % name)
        if speed_multiple > 1:
            fig.savefig('pdf/%s-fast.png' % name)
        else:
            fig.savefig('pdf/%s-full.png' % name)

        if 'SUPPRESS_PLOT' not in locals() or not SUPPRESS_PLOT:
            pl.show()


