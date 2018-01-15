'''
Created on 28 Dec 2017

@author: Alpha Renner

This class is a 2d plotter and provides functionality
for analysis of 2d neuron fields
To be extended!
'''

################################################################################################
# Import required packages
import csv
from brian2 import ms, defaultclock, second
import numpy as np
#import matplotlib.animation as animation
import pyqtgraph as pg
import sparse
from scipy import ndimage


class Plotter2d(object):
    """
    This is not so efficient, if get_filtered is called a lot, as the result is not stored
    """

    def __init__(self, monitor, rows, cols):

        self.rows = rows
        self.cols = cols
        self.mask = [True] * (len(monitor.t))
        self._t = monitor.t  # times of spikes
        self._i = monitor.i  # neuron index number of spike
        #print(self._i)
        self._xi, self._yi = np.unravel_index(self._i, (cols, rows))
        assert(len(self._i) == len(self._t))
        self.shape = (rows,cols,len(monitor.t))

    @property
    def t(self):
        return self._t[self.mask]

    @property
    def i(self):
        return self._i[self.mask]

    @property
    def xi(self):
        return self._xi[self.mask]

    @property
    def yi(self):
        return self._yi[self.mask]

    def set_range(self, mon_range):
        self.mask = (self._t <= mon_range[1]) & (self._t >= mon_range[0])

    def get_sparse3d(self, dt):
        sparse_spikemat = sparse.COO((np.ones(len(self.t)), (self.t / dt, self.xi, self.yi)),
                                     shape=(1 + int(max(self.t) / dt), self.cols, self.rows))
        return sparse_spikemat

    def get_dense3d(self, dt):
        sparse3d = self.get_sparse3d(dt)
        return sparse3d.todense()

    def get_filtered(self, dt, filtersize):
        dense3d = self.get_dense3d(dt)
        return ndimage.uniform_filter1d(dense3d, size=int(filtersize / dt), axis=0, mode='constant') * second / dt
    #    import timeit
    #    timeit.timeit("ndimage.uniform_filter(dense3d, size=(0,0,10))",
    #                  setup = 'from scipy import ndimage',
    #                  globals={'dense3d':dense3d},number = 1)
    #    timeit.timeit("ndimage.uniform_filter1d(dense3d,size=10, axis = 2, mode='constant')",
    #                  setup = 'from scipy import ndimage',
    #                  globals={'dense3d':dense3d},number = 1)
    #    timeit.timeit("ndimage.convolve1d(dense3d, weights=np.ones(10), axis = 2)",
    #                  setup = 'from scipy import ndimage;import numpy as np',
    #                  globals={'dense3d':dense3d},number = 1)

    def plot3d(self, plot_dt=defaultclock.dt, filtersize=10 * ms):

        video_filtered = self.get_filtered(plot_dt, filtersize)
        imv = pg.ImageView()
        imv.setImage(video_filtered, xvals=np.arange(
            0, video_filtered.shape[0] * (plot_dt / ms), plot_dt / ms))
        imv.setPredefinedGradient("thermal")
        # imv.show()
        # imv.export("plot/plot_.png")
        return imv

    def savez(self,filename):
        np.savez_compressed(str(filename)+".npz",self.i,self.t,self.rows,self.cols)

    @classmethod
    def loadz(cls,filename):
        with np.load(str(filename)) as loaded_npz:
            i,t,rows,cols = [loaded_npz[arr] for arr in loaded_npz]
        mon = lambda:0
        mon.t = t*second
        mon.i = i
        return cls(mon,rows,cols)

    def savecsv(self,filename):
        """
        not tested
        """
        with open(str(filename)+'.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.t/second)
            csvwriter.writerow(self.i)
