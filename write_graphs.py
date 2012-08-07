#!/usr/bin/env python
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import jsondata

if __name__=='__main__':
    rc('text', usetex=True)

    data = jsondata.read('table.json')

    pps = [0.1, 0.5, 0.9,]

    dd = [('generate_well_separable', 'Well Separated'),
          ('generate_mostly_separable', 'Mostly Separated'), 
          ('generate_some_overlap', 'Some Overlap'), 
          ('generate_complete_overlap', 'Complete Overlap'), 
    ]

    for distributions,name in dd:
        fig, axs = plt.subplots(3, sharex=True, sharey=True)

        # Three subplots sharing both x/y axes
        for i,pp in enumerate(pps):
            rows = [r[2:] for r in data if r[0] == pp and r[1] == distributions]
            d = np.array(rows).T
            d.sort(axis=1)
            assert d.shape == (6, 4) 

            xaxis = np.arange(0, 1.1, 0.2)[1:-1]
            axs[i].plot(xaxis, d[0], 'ko-', xaxis, d[0], 'k-')
            print d[0]

            #TODO: jperla: for i in xrange(zip)...
            colors = ['r', 'g', 'b', 'm', 'c']
            estimators = ['$e_1$', '$e_2$', '$e_3$', '$\hat{e_1}$', '$\hat{e_4}$']
            for j,(e, c) in enumerate(zip(estimators, colors)):
                axs[i].set_title('%i pos / %i neg' % (pp * 10000, ((1 - pp) * 10) * 1000))
                axs[i].plot(xaxis, d[j+1], '%so--' % c, label=e)

        handles, labels = axs[-1].get_legend_handles_labels()
        axs[-1].legend(handles, labels)

        fig.suptitle(name, fontsize=20, fontweight='bold')

        # Fine-tune figure; make subplots close to each other and hide x ticks for
        # all but bottom plot.
        #fig.subplots_adjust(hspace=0)
        plt.axis([0.0,1.0,-0.1,1.1])

        #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.xticks([0.2, 0.4, 0.6, 0.8], ['0.01', '0.1', '0.5', '0.9'], size='small')


        fig.savefig('graphs/%s.png' % name)
        
