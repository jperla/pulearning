import os
import logging
import functools

import scipy
import sklearn
import numpy as np

import sgdlr
import logistic
from fastgridsearch import FastGridSearchCV

#C_VALUES = [0.01, 0.1]
#C_VALUES = [1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2, 1e5, 1e8, 1e11, 1e14]
#C_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
C_VALUES = [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,]

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

def svm_label_data(train_set, train_labels, test_set, C=C_VALUES, CP=None, sample_weight=None):
    """Tries out several values for C by training on 75% of the training data, and validating on the rest.
        Picks the best values of C then uses it on the test set.
    """
    svms = []
    for c in C:
        cps = [c] if CP is None else CP
        for cp in cps:
            svm, accuracies = None, []

            svm = sklearn.svm.LinearSVC(C=c, class_weight={0: 1.0, 1: cp / c})
            svm.probability = True
            scores = sklearn.cross_validation.cross_val_score(svm, train_set, train_labels, 
                                                              cv=3, n_jobs=-1,
                                                              #scoring='roc_auc',
                                       )#                       fit_params={'sample_weight': sample_weight})
            accuracy = scores.mean()

            svms.append((accuracy, 1.0 / c, c, cp, svm))
            logging.debug('svm C=%s, CP=%s: %.2f%%' % (c, cp, 100 * accuracy))

    # get the top SVM
    best_svm = list(reversed(sorted(svms)))[0]
    logging.info('Best SVM: C=%s, CP=%s, accuracy=%.2f' % (best_svm[2], best_svm[3], best_svm[0]))

    svm = sklearn.svm.SVC(kernel='linear', cache_size=2000, C=best_svm[2], class_weight={0: 1.0, 1: best_svm[3] / best_svm[2]})
    svm.probability = True

    svm.fit(train_set, train_labels, sample_weight=sample_weight)
    svm_labels = svm.predict_proba(test_set)
    return svm_labels

if __name__=='__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    max_key = 24081

    pos, neg, unlabeled_pos = read_swissprot_data()
    # switch cases to use a smaller labeled dataset
    if 'switch_cases' in locals() and switch_cases:
        logging.warning('Switching positive and negative datasets!')
        unlabeled_pos, pos = pos, unlabeled_pos 

    truncate = lambda m: m[:int(m.shape[0] / 30),:]
    # Use less data so that we can move faster, comment this out to use full dataset
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
        X = scipy.sparse.csr_matrix(X) # sparsify X

        roc_curves = []

        # sci-kit learn's sgd classifier
        name = 'Sci-Kit SGD LR L2-regularized pos-only labels'
        logging.info('starting %s on pos-only data...' % name)
        sgd = sklearn.linear_model.SGDClassifier(loss='log', 
                                                penalty='l2', 
                                                alpha=0.001,
                                                random_state=2,
                                                n_iter=200)
        sgd.fit(X, y)
        sgd_labels = sgd.predict_proba(test_set)[:,1]
        logging.info('done %s: %s' % (name, np.max(np.abs(sgd.coef_))))
        fpr, tpr, roc_auc = calculate_test_roc(sgd_labels)
        roc_curves.append((name, roc_auc, fpr, tpr, 'p-'))
        logging.info('AUC for %s: %f' % (name, roc_auc))

        name = 'Sci-Kit SGD LR L2-regularized true labels'
        logging.info('starting %s on pos-only data...' % name)
        sgd = sklearn.linear_model.SGDClassifier(loss='log', 
                                                penalty='l2', 
                                                alpha=0.001,
                                                random_state=2,
                                                n_iter=200)
        sgd.fit(X, y_labeled)
        sgd_labels = sgd.predict_proba(test_set)[:,1]
        logging.info('done %s: %s' % (name, np.max(np.abs(sgd.coef_))))
        fpr, tpr, roc_auc = calculate_test_roc(sgd_labels)
        roc_curves.append((name, roc_auc, fpr, tpr, 'p-'))
        logging.info('AUC for %s: %f' % (name, roc_auc))

        # Baseline if we knew everything
        max_iter = 10
        logging.info('starting modified LR on pos-only data...')
        mlr = sgdlr.SGDModifiedLogisticRegression(alpha=0.01, n_iter=max_iter)
        mlr.fit(X, y)
        logging.info('done training Modified LR on pos-only data: %s' % (np.max(np.abs(mlr.theta_))))
        modified_regression_labels = mlr.predict_proba(test_set)

        logging.info('b = %s' % mlr.b_)
        logging.info('1.0 / (1.0 + b*b) = %s' % (1.0 / (1.0 + mlr.b_**2)))
        # Compute ROC curve and area the curve
        fpr, tpr, roc_auc = calculate_test_roc(modified_regression_labels)
        name = 'Modified LR pos-only labels'
        roc_curves.append((name, roc_auc, fpr, tpr, 'r--'))
        logging.info('AUC for %s: %f' % (name, roc_auc))


        lr = sgdlr.SGDLogisticRegression(alpha=0.01, n_iter=max_iter)

        logging.info('starting LR on totally labeled data...')
        lr.fit(X, y_labeled)
        logging.info('done LR')
        baseline_labels = lr.predict_proba(test_set)

        # calculate the parameters
        logging.info('starting LR on pos-only data...')
        lr.fit(X, y)
        logging.info('done LR')
        regression_labels = lr.predict_proba(test_set)

        # Compute ROC curve and area the curve
        fpr, tpr, roc_auc = calculate_test_roc(baseline_labels)
        name = 'LR true labels'
        roc_curves.append((name, roc_auc, fpr, tpr, 'r-'))
        logging.info('AUC for %s: %f' % (name, roc_auc))

        # Compute ROC curve and area the curve
        fpr, tpr, roc_auc = calculate_test_roc(regression_labels)
        name = 'LR pos-only labels'
        roc_curves.append((name, roc_auc, fpr, tpr, 'r-.'))
        logging.info('AUC for %s: %f' % (name, roc_auc))

        calculate_svms = True
        if calculate_svms:
            param_grid = {'C': C_VALUES}
            svm = FastGridSearchCV(sklearn.svm.LinearSVC(),
                                   sklearn.svm.SVC(probability=True),
                                   param_grid,
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=3)
            logging.info('starting SVM on pos-only data...')
            svm.fit(X, y)
            svm_labels = svm.predict_proba(test_set)
            fpr, tpr, roc_auc = calculate_test_roc(svm_labels[:,1])
            name = 'SVM pos-only labels'
            roc_curves.append((name, roc_auc, fpr, tpr, 'b-.'))
            logging.info('AUC for %s: %f' % (name, roc_auc))

            logging.info('starting Biased SVM...')
            svm_labels = svm_label_data(X, y, test_set, CP=C_VALUES)
            fpr, tpr, roc_auc = calculate_test_roc(svm_labels[:,1])
            name = 'Biased SVM pos-only labels'
            roc_curves.append((name, roc_auc, fpr, tpr, 'g-'))
            logging.info('AUC for %s: %f' % (name, roc_auc))

            logging.info('starting weighted SVM...')
            # I have to copy to avoid an error that says that svm_weight is not C-contiguous!
            svm_weight = svm_label_data(X, y, X)[:,1].copy()
            # now run this again with the probabilites as the weights
            svm_labels = svm_label_data(X, y, test_set, sample_weight=svm_weight, C=[0.125, 0.25, 0.5, 1.0])
            fpr, tpr, roc_auc = calculate_test_roc(svm_labels[:,1])
            name = 'Weighted SVM pos-only labels'
            roc_curves.append((name, roc_auc, fpr, tpr, 'g--'))
            logging.info('AUC for %s: %f' % (name, roc_auc))

            logging.info('starting SVM on true labels...')
            svm_labels = svm_label_data(X, y_labeled, test_set)
            fpr, tpr, roc_auc = calculate_test_roc(svm_labels[:,1])
            name = 'SVM true labels'
            roc_curves.append((name, roc_auc, fpr, tpr, 'b-'))
            logging.info('AUC for %s: %f' % (name, roc_auc))

        # Plot ROC curve
        import pylab as pl
        pl.clf()

        sorted_roc_curves = list(reversed(sorted(roc_curves, key=lambda a: a[1])))
        for (name, roc_auc, fpr, tpr, color) in sorted_roc_curves:
            pl.plot(fpr, tpr, color, label='%s (AUC = %0.4f)' % (name, roc_auc))

        pl.plot([0, 1], [0, 1], 'k--')
        if sorted_roc_curves[-1][1] < 0.8:
            # show the full curve if there are low roc_auc
            pl.xlim([0.0, 1.0])
            pl.ylim([0.0, 1.0])
        else:
            pl.xlim([0.0, 0.3])
            pl.ylim([0.7, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC for Protein Datasets')
        pl.legend(loc="lower right")
        pl.show()


