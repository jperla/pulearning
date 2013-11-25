import os
import logging

import numpy as np
import sklearn

import logistic

def read_swissprot_data():
    """Reads in swissprot dataset from 3 files in proteindata folder.
        Returns 3-tuple of numpy arrays.
    """
    folder = 'proteindata'
    filenames = ['P', 'N', 'Q']

    npy_filenames = 'pos', 'neg', 'test_pos'
    return (np.load(os.path.join(folder, 'data.%s.swissprot.npy' % d)) for d in npy_filenames)

if __name__=='__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    max_key = 24081

    pos, neg, unlabeled_pos = read_swissprot_data()

    # Use less data so that we can move faster, comment this out to use full dataset
    #truncate = lambda m: m[:int(m.shape[0] / 30),:]
    #pos, neg, unlabeled_pos = truncate(pos), truncate(neg), truncate(unlabeled_pos)

    num_folds = 10
    kfold_pos = list(sklearn.cross_validation.KFold(pos.shape[0], k=num_folds, shuffle=True, random_state=0))
    kfold_neg = list(sklearn.cross_validation.KFold(neg.shape[0], k=num_folds, shuffle=True, random_state=0))
    kfold_unlabeled_pos = list(sklearn.cross_validation.KFold(unlabeled_pos.shape[0], k=num_folds, shuffle=True, random_state=0))

    for i in range(1):
        pos_indices_train, pos_indices_test = kfold_pos[i]
        neg_indices_train, neg_indices_test = kfold_neg[i]
        unlabeled_pos_indices_train, unlabeled_pos_indices_test = kfold_unlabeled_pos[i]

        pos_train, pos_test = pos[pos_indices_train], pos[pos_indices_test]
        neg_train, neg_test = neg[neg_indices_train], neg[neg_indices_test]
        unlabeled_pos_train, unlabeled_pos_test = unlabeled_pos[unlabeled_pos_indices_train], unlabeled_pos[unlabeled_pos_indices_test]

        test_set = np.vstack([pos_test, unlabeled_pos_test, neg_test])
        test_labels = np.hstack([np.array([1] * (pos_test.shape[0] + unlabeled_pos_test.shape[0])),
                                 np.array([0] * neg_test.shape[0])])

        # set up the datasets
        X = np.vstack([pos_train, neg_train, unlabeled_pos_train])
        y = np.hstack([np.array([1] * pos_train.shape[0]),
                      np.array([0] * (neg_train.shape[0] + unlabeled_pos_train.shape[0]))])
        X, y = sklearn.utils.shuffle(X, y)

        # calculate the parameters
        max_iter = 100
        logging.info('starting LR...')
        thetaR = logistic.fast_logistic_gradient_descent(X,
                                                         y, 
                                                         max_iter=max_iter)
        logging.info('done LR...')
        logging.info('starting modified LR...')
        thetaMR, b = logistic.fast_modified_logistic_gradient_descent(X,
                                                                      y, 
                                                                      max_iter=max_iter, 
                                                                      alpha=0.01)
        logging.info('done modified LR')

        # label the test set
        regression_labels = logistic.label_data(test_set, thetaR, binarize=False)
        modified_regression_labels = logistic.label_data(test_set, thetaMR, (b * b), binarize=False)

        print regression_labels
        print modified_regression_labels

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_labels, regression_labels)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print("Area under the ROC curve for standard logistic regression: %f" % roc_auc)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_labels, modified_regression_labels)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print("Area under the ROC curve for modified logistic regression: %f" % roc_auc)

        # Plot ROC curve
        import pylab as pl
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        pl.show()

