#!/usr/bin/env python
import random

from matplotlib import pyplot
import sklearn
import sklearn.feature_extraction
import sklearn.dummy
import numpy as np
import scipy

np.seterr(all='raise')

import lr
import logistic


if __name__ == '__main__':
    FULL_GRAPH = True
    if 'FULL_GRAPH' in locals() and FULL_GRAPH:
        speed_multiple = 1
    else:
        speed_multiple = 10

    import csv

    def unicode_csv_reader(utf8_data, dialect=csv.excel, limit=None, **kwargs):
        csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
        for i,row in enumerate(csv_reader):
            yield [unicode(cell, 'utf-8') for cell in row]
            if limit is not None and i > limit:
                break

    filename = 'nytimes-smiles/results.csv'
    
    line_limit = 30000
    if speed_multiple > 1:
        line_limit = 200


    reader = unicode_csv_reader(open(filename), limit=line_limit)
    lines = [f for f in reader][1:]
    if speed_multiple > 1:
        lines = lines[:line_limit]
    raw_text = [l[30] for l in lines if l[32] != u'']
    raw_labels = [(float(l[32]), random.random()) for l in lines if l[32] != u'']

    counter = sklearn.feature_extraction.text.CountVectorizer()
    counts = counter.fit_transform([t for t in raw_text]).todense()
    # overfitting, so cut out a bunch of features
    counts = counts[:,:10]

    optimal_points = []
    posonly_points = []
    naive_points = []
    
    if speed_multiple > 1:
        cs = [(0.1 * i) for i in xrange(1, 10)]
    else:
        cs = [(0.01 * i) for i in xrange(1, 10)]
        '''
        cs.extend([(0.99 + (0.001 * i)) for i in xrange(1, 10)])
        cs.extend([(0.001 * i) for i in xrange(1, 10)])
        '''

    for c in cs:
        positive = np.vstack([d for i,d in enumerate(counts) if raw_labels[i][0] > 0])
        negative = np.vstack([d for i,d in enumerate(counts) if raw_labels[i][0] < 0])
        positive_labeled = np.vstack([d for i,d in enumerate(counts) if raw_labels[i][0] > 0 and raw_labels[i][1] < c])

        #skip if too high
        if positive_labeled.shape[0] == positive.shape[0]:
            continue

        positive_unlabeled = np.vstack([d for i,d in enumerate(counts) if raw_labels[i][0] > 0 and raw_labels[i][1] >= c])
        unlabeled = np.vstack([positive_unlabeled, negative])

        print positive.shape
        print negative.shape
        print positive_labeled.shape
        print positive_unlabeled.shape
        
        # skip data with no positive labels
        if positive_labeled.shape[0] < 3:
            continue

        X = np.vstack([positive_labeled, unlabeled])
        y = np.hstack([np.array([1] * positive_labeled.shape[0]),
                       np.array([0] * unlabeled.shape[0]),])
        X, y = sklearn.utils.shuffle(X, y)
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        scaler.fit(X)
        X = scaler.transform(X)
        X = scipy.sparse.csr_matrix(X)

        testX = np.vstack([positive, negative])
        testY = np.hstack([np.array([1] * positive.shape[0]),
                           np.array([0] * negative.shape[0]),])
        testX, testY = sklearn.utils.shuffle(testX, testY)
        testX = scaler.transform(testX)
        testX = scipy.sparse.csr_matrix(testX)

        optimalTrainX, optimalTrainY = testX, testY

        print 'c:', c

        n_iter = 1000

        posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=n_iter, eta0=0.1, c=c)
        posonly.fit(X, y)
        t = posonly.score(testX, testY)
        #t = sklearn.metrics.roc_auc_score(testY, posonly.predict_proba(testX)[:,1])
        posonly_points.append([c, t])
        print 'posonly:', t, 'c:', posonly.final_c()

        sgd_params = {'alpha':[1e-100, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                      'loss':['log', 'hinge'],
                      'penalty':['l2', 'l1'],
                     }
        true_sgd = sklearn.linear_model.SGDClassifier(loss='log', alpha=1e-3)
        #true_sgd = sklearn.grid_search.GridSearchCV(true_sgd, sgd_params)
        true_sgd.fit(optimalTrainX, optimalTrainY)
        #print true_sgd.best_params_
        t = true_sgd.score(testX, testY)
        #t = sklearn.metrics.roc_auc_score(testY, true_sgd.predict_proba(testX)[:,1])
        optimal_points.append([c, t])
        print 'maximum:', t

        sgd = sklearn.linear_model.SGDClassifier(loss='log', alpha=1e-100)
        sgd = sklearn.grid_search.GridSearchCV(sgd, sgd_params)
        sgd.fit(X, y)
        #print sgd.best_params_
        t = sgd.score(testX, testY)
        '''
        try:
            t = sklearn.metrics.roc_auc_score(testY, sgd.predict_proba(testX)[:,1])
        except:
            t = 0.0
        '''
        naive_points.append([c, t])
        print 'naive sgd:', t

    optimal_points = np.array(sorted(optimal_points))
    posonly_points = np.array(sorted(posonly_points))
    naive_points = np.array(sorted(naive_points))

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(posonly_points[:,0], posonly_points[:,1], 'b+--', label="POLR pos-only labels")
    ax.plot(optimal_points[:,0], optimal_points[:,1], 'go-', label="LR true labels")
    ax.plot(naive_points[:,0], naive_points[:,1], 'rx-', label="LR pos-only labels")

    ax.set_title('Comparing logistic regression on synthetic data')
    ax.set_xlabel('C')
    ax.set_ylabel('Test Accuracy')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)

    name = 'sentiment'
    fig.savefig('pdf/%s.png' % name)
    if speed_multiple > 1:
        fig.savefig('pdf/%s-fast.png' % name)
    else:
        fig.savefig('pdf/%s-full.png' % name)

    if 'SUPPRESS_PLOT' not in locals() or not SUPPRESS_PLOT:
        fig.show()

