'''
Created on 28 Dec 2017

@author: Alpha Renner

This class is a 2d plotter and provides functionality
for analysis of 2d neuron fields
To be extended!

Attributes:
    CM_JET (TYPE): Description
    CM_ONOFF (TYPE): Description
'''

################################################################################################
# Import required packages
import csv
import os
import sys
from brian2 import ms, Hz, defaultclock, second
import numpy as np
import shutil
# import matplotlib.animation as animation
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui  # , QtCore
import pyqtgraph.exporters  # looks redundant, but this is necessary for export
import sparse
from scipy import ndimage
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
# pg.setConfigOption('background', 'w') # makes  background white
from pyqtgraph.colormap import ColorMap

CM_JET = ColorMap([0.0, 0.33, 0.66, 1.0],
                  [(0, 0, 255, 255), (0, 255, 255, 255),
                   (255, 255, 0, 255), (255, 10, 10, 255)], mode=2)

CM_ONOFF = ColorMap([0.0, 0.33, 0.66, 1.0],
                    [(0, 0, 0, 255), (0, 255, 0, 255),
                     (255, 0, 0, 255), (255, 255, 0, 255)], mode=2)

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))


class DVSmonitor:
    """Summary

    Attributes:
        pol (TYPE): Description
        t (TYPE): Description
        xi (TYPE): Description
        yi (TYPE): Description
    """

    def __init__(self, xi, yi, t, pol, unit=None):
        """Summary

        Args:
            xi (TYPE): Description
            yi (TYPE): Description
            t (TYPE): Description
            pol (TYPE): Description
        """
        if unit is not None:
            self.t = np.asarray(t) * unit
        else:
            try:
                if t.dim == second.dim:
                    self.t = t
                else:
                    # this means it has a brian2 dim that is not second
                    # or it is of some other type that has a .dim!
                    raise Exception('t does not have time as dimension/unit')
            except AttributeError:
                self.t = np.asarray(t) * ms

        self.xi = xi
        self.yi = yi
        self.pol = pol


class Plotter2d(object):
    """
    Plotter2d is a class that contains a number of functions to create 2d plots over time, in particular events/spikes
     that are arranged in 2d such as DVS camera recordings or 2d neural fields.
     The class offers filtering, plotting and generation of gifs.

     Data is passed into the plotter as a monitor (either from brian2 or using the DVSmonitor class)

    Attributes:
        dims (tuple): the dimensions of the 2d data (number of rows and columns)
        monitor (TYPE): A monitor to sparsely store event data (e.g. from brian2),
            it has a t (timestamps), xi (x event coordinates) and
            yi (y coordinates) property or an i property (flat coordinates that are reshaped to 2d)
        plotrange (tuple): Masks the monitor outside of the given range (in units of t)
        shape (tuple): 3d shape of the data
        rows (int): number of rows (dims[0])
        cols (int): number of columns (dims[1])
        mask (array): mask that masks out part of the data
    """

    def __init__(self, monitor, dims, plotrange=None):
        """Summary

        Args:
            monitor (TYPE): A monitor, that has a t (timestamps), xi (x coordinates) and yi (y coordinates) property
            dims (tuple): the dimensions of the 2d video that schould be plotted
            plotrange (tuple): Masks the monitor outside of the given range (in units of t)
        """
        self.rows = dims[0]
        self.cols = dims[1]
        self.dims = dims

        self._t = monitor.t  # times of spikes
        self.shape = (dims[0], dims[1], len(monitor.t))

        self.monitor = monitor  # mainly for debugging!

        # self.name = monitor.name
        try:  # that should work if the monitor is a Brian2 Spikemonitor
            self._i = monitor.i  # neuron index number of spike
            # print(self._i)
            self._xi, self._yi = np.unravel_index(self._i, (dims[0], dims[1]))
            # assert(len(self._i) == len(self._t))
        except ValueError as e:
            print('You probably did not set the correct dimensions for your input!')
            raise e
        except AttributeError:  # that should work, if it is a DVSmonitor (it has xi and yi instead of y)
            self._xi = np.asarray(monitor.xi, dtype='int')
            self._yi = np.asarray(monitor.yi, dtype='int')
            self._i = np.ravel_multi_index((self._xi, self._yi), dims)  # neuron index number of spike
            try:  # check, if _t has a unit (dvs raw data is given in ms)
                self._t[0].dim
            except:
                self._t = self._t * ms

        try:
            self._pol = monitor.pol
        except:
            self._pol = np.zeros_like(self._i)

        self.mask = range(len(monitor.t))  # [True] * (len(monitor.t))

        self._plotrange = (0 * ms, 0 * ms)
        self.set_range(plotrange)

    @property
    def plotrange(self):
        return self._plotrange

    @plotrange.setter
    def plotrange(self, plotrange):
        self.set_range(plotrange)

    @property
    def pol(self):
        """polarity of DVS spikes
        """
        return self._pol[self.mask]

    @property
    def t(self):
        """timestamps of events
        """
        return self._t[self.mask]

    @property
    def t_(self):
        """
        unitless t in ms
        """
        return self._t[self.mask] / ms

    @property
    def i(self):
        """flattened indices (in 1d)
        """
        return self._i[self.mask]

    @property
    def xi(self):
        """row coordinates of events
        """
        return self._xi[self.mask]

    @property
    def yi(self):
        """columns coordinates of events
        """
        return self._yi[self.mask]

    @property
    def plotlength(self):
        """number of timesteps
        """
        # if self.plotrange is not None:
        plotlength = self.plotrange[1] - self.plotrange[0]
        # else:
        #    plotlength = np.max(self.t)
        return plotlength

    def plotshape(self, dt):
        """3d shape of the data (num_timestamps, num_rows, num_cols)

        Args:
            dt (float): timestep length

        Returns:
            tuple: (num_timestamps, num_rows, num_cols)
        """
        plottimesteps = int(np.ceil(0.0001 + self.plotlength / dt))
        # print(plottimesteps)
        return (plottimesteps, self.dims[0], self.dims[1])

    def set_range(self, plotrange=None):
        '''
        set a range with unit that is applied for all computations with this monitor

        Args:
            plotrange (tuple): (from, to))
        '''
        if plotrange is None:
            self.mask = range(len(self._t))  # slice(len(self._t))  # [True] * (len(self._t))
            if len(self.t) > 0:
                self._plotrange = (np.min(self.t), np.max(self.t))
            else:
                self._plotrange = (0 * ms, 0 * ms)
        else:
            self._plotrange = plotrange
            self.mask = np.where((self._t <= plotrange[1]) & (self._t >= plotrange[0]))[0]

    def get_sparse3d(self, dt, align_to_min_t=True):
        """Using the package sparse (based of scipy sparse, but for 3d), the spiketimes
        are converted into a sparse matrix. This step is basically just for easy
        conversion into a dense matrix later, as you cannot do so many computations
        with the sparse representation.
        sparse documentation can be found here: http://sparse.pydata.org/en/latest/

        Args:
            dt (float): the t dimension in the sparse representation is given in timesteps, so t is divided by dt

        Returns:
            sparse.COO: Sparse representation of the data used to create the dense one efficiently.
        """
        # print(len(self.t))
        # print(np.max(self.t / dt))
        # print(self.plotshape(dt))
        if len(self.t) > 0:
            if align_to_min_t:
                min_t = np.min(self.t)
            else:
                min_t = 0 * ms

            try:
                sparse_spikemat = sparse.COO(
                    (np.ones(len(self.t)), ((self.t - min_t) / dt, self.xi, self.yi)),
                    shape=self.plotshape(dt))
            except:
                sparse_spikemat = sparse.COO(
                    coords=((self.t - min_t) / dt, self.xi, self.yi),
                    data=np.ones(len(self.t)),
                    shape=self.plotshape(dt))
        else:
            print('Your monitor is empty!')
            # just create a matrix of zeros, hope, this does not lead to other problems
            sparse_spikemat = sparse.COO(([0], ([0], [0], [0])), shape=self.plotshape(dt))
        return sparse_spikemat

    # Example:
    # sparse_test = sparse.COO((np.ones(5), (np.asarray([0,10,40,60,80]) / 10, [1,2,3,4,5], [5,4,3,2,1])),shape=(9,6,6))
    # print(sparse_test.todense())

    def get_dense3d(self, dt):
        """Transforms the sparse spike time representation in a dense representation,
        where every spike is given as a 1 in a 3d matrix (time + 2 spatial dimensions)
        The data is binned using dt. If there is more than one spike in a bin, the bin will
        not have the value 1, but the number of spikes.

        Args:
            dt (TYPE): Description

        Returns:
            array: dense array of the data. E.g. if we have a single spike in the neuron at location (3, 5) at timestamp
                10, in the dense array, all locations have a value of 0 at all timesteps apart from (10, 3, 5), where
                the value is 1.
        """
        sparse3d = self.get_sparse3d(dt)
        return sparse3d.todense()

    def get_filtered(self, dt, filtersize):
        """applies a rectangular filter (convolution) of length filtersize over time (dimension 0).
        It returns a 3d matrix with the firing rate.
        Spiketimes will be binned with a step size of dt that means that the filtersize should always be a int multiple of dt

        Args:
            dt (brian2.Quantity): the time step with which the spike times are binned
            filtersize (brian2.Quantity): length of the filter (in brian2 time units)
        Returns:
            TYPE: Description
        """
        dense3d = self.get_dense3d(dt)
        filtered = ndimage.uniform_filter1d(dense3d, size=int(filtersize / dt),
                                            axis=0, mode='constant') * second / dt
        # filtered  = ndimage.zoom(filtered, (1, 2, 2))
        return filtered

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

    def plot3d_on_off(self, plot_dt=defaultclock.dt, filtersize=10 * ms, colormap=CM_ONOFF, flipy= False):
        """
        Args:
            plot_dt (brian2.Quantity, optional): timestep in which events are binned for plotting
            filtersize (brian2.Quantity, optional): filtersize of rectangular filter
            colormap (pyqtgraph.colormap.ColorMap, optional): colormap for on off plot

        Returns:
            pyqtgraph.ImageView: ImageView object for usage in a larger pyqtgraph plot
        """

        video_filtered0 = 0
        video_filtered1 = 0

        if self._pol is None:
            print('no polarity information stored, cannot create on-off plot')
            return

        time_mask = self.mask
        pol_mask = np.where((self._pol == 0))[0]
        self.mask = np.sort(np.asarray(list(set(time_mask).intersection(pol_mask)), dtype=int))

        if len(self.t) > 0:
            try:
                video_filtered0 = self.get_filtered(plot_dt, filtersize)
            except MemoryError:
                raise MemoryError("the dt you have set would generate a too large matrix for your memory")
                video_filtered0 = self.get_filtered(plot_dt * 10, filtersize)
            video_filtered0[video_filtered0 > 0] = 1

        pol_mask = np.where((self._pol == 1))[0]
        self.mask = np.sort(np.asarray(list(set(time_mask).intersection(pol_mask)), dtype=int))
        if len(self.t) > 0:
            try:
                video_filtered1 = self.get_filtered(plot_dt, filtersize)
            except MemoryError:
                raise MemoryError("the dt you have set would generate a too large matrix for your memory")
                video_filtered1 = self.get_filtered(plot_dt * 10, filtersize)
            video_filtered1[video_filtered1 > 0] = 2

        video_filtered = video_filtered0 + video_filtered1

        self.mask = time_mask

        if flipy:
            video_filtered = np.flip(video_filtered, 2)

        imv = pg.ImageView()
        imv.setImage(video_filtered, xvals=np.min(self.t / ms) + np.arange(
            0, video_filtered.shape[0] * (plot_dt / ms), plot_dt / ms))
        imv.ui.histogram.gradient.setColorMap(colormap)
        # imv.setPredefinedGradient("thermal")
        # imv.show()
        # imv.export("plot/plot_.png")

        return imv

    def plot3d(self, plot_dt=defaultclock.dt, filtersize=10 * ms, colormap=CM_JET, levels=None, flipy= False):
        """
        Args:
            plot_dt (brian2.Quantity, optional): timestep in which events are binned for plotting
            filtersize (brian2.Quantity, optional): filtersize of rectangular filter
            colormap (pyqtgraph.colormap.ColorMap, optional): colormap for on off plot
            levels (tuple): (min, max); the white and black level values to use (passed to pyqtgraph)

        Returns:
            pyqtgraph.ImageView: ImageView object for usage in a larger pyqtgraph plot
        """
        try:
            video_filtered = self.get_filtered(plot_dt, filtersize)
        except MemoryError:
            raise MemoryError("the dt you have set would generate a too large matrix for your memory")

        if flipy:
            video_filtered = np.flip(video_filtered, 2)

        imv = pg.ImageView()
        imv.setImage(video_filtered, xvals=np.min(self.t / ms) + np.arange(
            0, video_filtered.shape[0] * (plot_dt / ms), plot_dt / ms), levels=levels)
        imv.ui.histogram.gradient.setColorMap(colormap)
        # imv.setPredefinedGradient("thermal")
        # imv.show()
        # imv.export("plot/plot_.png")
        return imv

    def rate_histogram(self, filename=None, filtersize=50 * ms, plot_dt=defaultclock.dt * 100, num_bins=50):
        """plots a histogram of rates

        Args:
            filename (str): filename to save the histogram
            filtersize (brian2.Quantity, optional): filtersize of the retangular filter to calculate the rate
            plot_dt (brian2.Quantity, optional): binsize in which the data is binned
            num_bins (int, optional): number of bins of the histogram
        """
        video_filtered = self.get_filtered(plot_dt, filtersize)
        histrange = (0, np.max(video_filtered))
        num_bins = num_bins
        flat_rate_time = np.reshape(
            video_filtered, (video_filtered.shape[0], video_filtered.shape[1] * video_filtered.shape[2]))
        hist2d = np.zeros((len(flat_rate_time), num_bins))
        for t in range(len(flat_rate_time)):
            # ,density = True)
            hist = np.histogram(
                flat_rate_time[t], bins=num_bins, range=histrange)
            hist2d[t] = np.log10(hist[0])

        hist2d[hist2d == -np.inf] = 0

        hist2d = np.flip(hist2d, 1)
        densetimes = np.arange(
            self.plotrange[0] / ms, self.plotrange[1] / ms, plot_dt / ms)
        pddf_rate = pd.DataFrame(data=hist2d.T,  # values
                                 # 1st column as index
                                 index=np.flip(
                                     np.round(hist[1][0:(len(hist[1]) - 1)], 0), 0),
                                 columns=densetimes / 1000)

        plt.figure()
        sns_fig = sns.heatmap(pddf_rate, cmap='jet', vmax=None).get_figure()
        plt.xlabel('time in s')
        plt.ylabel('firing rate in Hz')
        # plt.show()
        if filename is not None:
            sns_fig.savefig(str(filename) + '_ratehistogram' + '.png')
            plt.close()
        # plt.figure()
        # plt.imshow(hist2d.T/np.max(hist2d))#, vmax = 0.1)

    def get_dense_ifr(self, dt=50 * ms, plot=False, frames_timestamps=None):
        """
        calculates a vector of instantaneous frequencies for every timestep dt.
        IFRs on timesteps without a spike are interpolated between the last two spikes
        :return: matrix of IFRs for every neuron and every timestep
        """

        from functools import partial
        import multiprocessing

        if frames_timestamps is None:
            densetimes = np.arange(
                self.plotrange[0] / ms, self.plotrange[1] / ms, dt / ms)
        else:
            densetimes = frames_timestamps / ms
            print('dt will be ignored as you have set frame_timestamps')

        # denseisis = np.zeros((len(densetimes), self.cols * self.rows))

        interp = partial(interpolate_isi, t=self.t_, i=self.i, densetimes=densetimes)

        # for i in range(self.cols * self.rows):
        #     interpolate_isi(i)
        #     denseisis[:, i] = interpolate_isi(i)
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores - 1)
        indices = range(0, self.cols * self.rows)
        res = pool.map(interp, indices)
        denseisis = np.vstack(res).T

        denseifrs = 1 / (denseisis / 1000)
        denseifrs[denseifrs == np.inf] = 0

        if plot:
            imv = pg.ImageView()
            # imv.setImage(np.reshape(1/(denseisis/1000),(denseisis.shape[0],self.cols,self.rows)))
            # imv.setImage(np.reshape(denseisis, (denseisis.shape[0], self.cols, self.rows)))
            imv.setImage(np.reshape(denseifrs, (denseifrs.shape[0], self.cols, self.rows)))
            imv.setPredefinedGradient("thermal")
            imv.show()
            # app.exec()

        return denseifrs, denseisis, densetimes

    def ifr_histogram(self, filename=None, num_bins=50):
        """histogram of instantaneous frequencies

        Args:
            filename (str): filename to save the histogram
            num_bins (int, optional): number of bins of the histogram
        """
        denseifrs, denseisis, densetimes = self.get_dense_ifr(dt=5 * ms)
        histrangeifr = (0, np.max(denseifrs))
        histrangeisi = (0, np.max(denseisis))
        histrangeisi = (0, 200)
        num_bins = num_bins
        hist2disi = np.zeros((len(denseisis), num_bins))
        hist2difr = np.zeros((len(denseifrs), num_bins))
        for t in range(len(denseisis)):
            # ,density = True)
            histisi = np.histogram(
                denseisis[t], bins=num_bins, range=histrangeisi)
            hist2disi[t] = histisi[0]
            # ,density = True)
            histifr = np.histogram(
                denseifrs[t], bins=num_bins, range=histrangeifr)
            hist2difr[t] = np.log10(histifr[0])

        hist2disi = np.flip(hist2disi, 1)
        hist2difr = np.flip(hist2difr, 1)

        hist2difr[hist2difr == -np.inf] = 0
        hist2difr[hist2difr == np.nan] = 0

        # Make pandas df with colnames and rownames for nicer plotting with sns
        pddf_ifr = pd.DataFrame(data=hist2difr.T,  # values
                                # 1st column as index
                                index=np.flip(
                                    np.round(histifr[1][0:(len(histifr[1]) - 1)], 0), 0),
                                columns=densetimes / 1000)

        plt.figure()
        sns_fig = sns.heatmap(pddf_ifr, cmap='jet', vmax=None,
                              cbar_kws={'label': 'log(n)'}).get_figure()
        # sns_fig = sns.heatmap(hist2difr.T,cmap = 'jet',xticklabels=densetimes,yticklabels=np.flip(np.round(histifr[1][0:(len(histifr[1])-1)],0),0),vmax=None).get_figure()
        plt.xlabel('time in s')
        plt.ylabel('ifr in Hz')
        if filename is None:
            plt.show()
        else:
            sns_fig.savefig(str(filename) + '_ifrhistogram' + '.png')
            plt.close()

    def savez(self, filename):
        """
        saves the object in a sparse way.
        only i,t, rows and cols are saved to an npz

        Args:
            filename (str): filename under which to save the data of the plotter object
        """
        np.savez_compressed(str(filename) + ".npz", self.i,
                            self.t, self.pol, self.dims)

    @classmethod
    def loadz(cls, filename):
        """
        loads a file that has previously been saved with savez and returns a
        SpikeMonitor2d object

        usage:
            spikemonObject = SpikeMonitor2d.loadz(myfilename)
            #e.g.
            spikemonObject.plot3d()

        Args:
            filename (str):  filename from where to load the data of the plotter object

        Returns:
            TYPE: Description
        """

        def mon():
            """Summary

            Returns:
                TYPE: Description
            """
            return 0

        try:
            with np.load(str(filename)) as loaded_npz:
                i, t, pol, dims = [loaded_npz[arr] for arr in loaded_npz]
            assert len(dims) == 2
            mon.t = t * second
            mon.i = i
            mon.pol = pol
            return cls(mon, dims)
        except:  # This is for backwards compatibility
            print("You are probably loading an old saved monitor!")
            try:
                with np.load(str(filename)) as loaded_npz:
                    i, t, dims = [loaded_npz[arr] for arr in loaded_npz]
                mon.t = t * second
                mon.i = i
                return cls(mon, dims)
            except:
                with np.load(str(filename)) as loaded_npz:
                    i, t, rows, cols = [loaded_npz[arr] for arr in loaded_npz]
                mon.t = t * second
                mon.i = i
                return cls(mon, (rows, cols))

    @classmethod
    def loaddvs(cls, eventsfile, dims=None):
        """
        loads a dvs numpy (events file) from aedat2numpy and returns a
        SpikeMonitor2d object, you can also directly pass an events array

        usage:
            spikemonObject = SpikeMonitor2d.loadz(myfilename)
            #e.g.
            spikemonObject.plot3d()

        Args:
            eventsfile (str):   filename from where to load the data of the plotter object
        """
        if type(eventsfile) == str:
            events = np.load(eventsfile)
        else:
            events = eventsfile
        mon = DVSmonitor(*list(events))
        if dims is None:
            dims = (int(1 + np.max(mon.xi)), int(np.max(1 + mon.yi)))
        return cls(mon, dims)

    def savecsv(self, filename):
        """
        export data as csv
        not tested

        Args:
            filename (TYPE): Description
        """
        with open(str(filename) + '.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.t / second)
            csvwriter.writerow(self.i)

    def plot_panes(self, num_panes=None, timestep=None, filtersize=50 * ms, num_rows=2,
                   plotfunction='plot3d', filename=None, colormap=cm.jet, **plotkwargs):
        """
        plots the 3d data as time slices (2d images at several timepoints)
        Args:
            num_panes (int, optional): number of panes to plot
            timestep (TYPE, optional): timestep between panes
            filtersize (TYPE, optional): filtersize at which the spikes are filtered to generate the images
            num_rows (int, optional): number of rows of the pane plot
            filename (str, optional): location where to save the plot

        Returns:
            TYPE: Description
        """

        if type(plotfunction) == str:
            plotfunction = getattr(self, plotfunction)

        if num_panes is None and timestep is None:
            print('please specify either num_panes or timestep')
            return
        if num_panes is not None and timestep is not None:
            print('please specify either num_panes or timestep, not both!')
            return
        if num_panes is not None:
            timestep = self.plotlength / num_panes

        dt = filtersize / 10
        num_steps = int(np.round(timestep / dt))

        video_filtered = plotfunction(plot_dt=dt, filtersize=filtersize, plot=False, **plotkwargs)
        # frames_per_row = num_panes//num_rows #int(len(video_filtered)/num_steps/num_rows)
        slice_indices = np.arange(0, num_panes * num_steps, num_steps)
        gw_paneplot = create_panes(video_filtered, num_rows, slice_indices, colormap=colormap)
        export_panes(gw_paneplot=gw_paneplot, filename=filename)
        return gw_paneplot

    def generate_movie(self, filename, scale=None, speed=1, plotfunction='plot3d',
                       plot_dt=10 * ms, tempfolder=os.path.expanduser('~'),
                       ffmpegoptions='', **plotkwargs):
        """
        This exports a movie or gif from an imageview
        Existing outputfiles will be overwritten
        This needs ffmpeg wich is installed on most linux distributions and also available for windows and mac
        Have a loo here: https://ffmpeg.org/

        Args:
            filename (str): The filename in which to store the generated movie.
                            You can choose a format that can be generated with ffmpeg like '.gif', '.mpg', '.mp4',...
            scale (str, optional): give pixel size as string e.g. '100x100'
            speed (num, optional): if the video should run faster, specify a multiplier
            plot_dt (given in brian2 time units): is passed to the plotfunction and determines the fps
            tempfolder (str, optional): the directory in which the temporary folder to store
                                        files created in the process.
                                        The temporary folder will be deleted afterwards.
                                        By default it will be created in your home directory
            plotfunction (str or function, optional): the function that should be used to create the gif.
                                                      it has to be a function that returns a pyqtgraph imageview
                                                      (or at least something similar that can export single images)
                                                      like the methods of this class (plot3d, ...). For the methods,
                                                      you can also pass a string to identify the plotfunction.
                                                      The plotfunction has to take plot_dt as an argument
            ffmepgoptions (str, optional):
            kwargs: all other keyword agruments will be passed to the plotfunction

            Example usage:
            plotter2dobject.generate_gif('~/gifname.gif', plotfunction = 'plot3d_on_off', filtersize=100 * ms, plot_dt=50 * ms)
        """
        desired_fps = 50
        fps = np.asarray(speed / plot_dt / Hz, dtype='int')  # theoretical framerate for dt
        pts = desired_fps / fps  # frames to drop in order to get actual framerate of 30 fps (presentation timestamp)
        # if abs(speed / plot_dt / Hz - fps) > 0.0000001:
        #     plot_dt = 1 / fps * second
        #     print('Your plot_dt was rounded to', plot_dt, 'in order to fit framerate of', fps)

        gif_temp_dir = os.path.join(tempfolder, "gif_temp")
        # pgImage = self.plot3d(plot_dt=plot_dt, filtersize=filtersize)
        if type(plotfunction) == str:
            plotfunction = getattr(self, plotfunction)
        pgImage = plotfunction(plot_dt=plot_dt, **plotkwargs)
        if not os.path.exists(gif_temp_dir):
            os.makedirs(gif_temp_dir)
        try:
            pgImage.export(os.path.join(gif_temp_dir, "gif.png"))
        except AttributeError as e:
            print(e)
            print('No gif created, probably empty monitor')
            return

        # before switching to ffmpeg we used convert, which is less flexible concerning framerates
        #        linux_command = "cd " + str(gif_temp_dir) + ";" + \
        #                "convert -delay "+str(delay)+" *.png "+ os.path.abspath(filename)

        if not '.' in filename:
            filename = filename + '.gif'

        ffmpeg_command = "cd " + \
                         str(gif_temp_dir) + ";" + \
                         "ffmpeg -f image2 -framerate " + str(desired_fps) + " -pattern_type glob -i '*.png' "
        # ffmpeg_command += "'setpts=" + str(pts) + "*PTS' "  # control speed

        if scale is not None:
            ffmpeg_command += "-filter_complex scale=" + scale + ",setpts=" + str(pts) + "*PTS  "
        else:
            ffmpeg_command += "-filter_complex setpts=" + str(pts) + "*PTS  "

        ffmpeg_command += '-y '  # overwrite existing output files
        ffmpeg_command += ffmpegoptions + ' '
        ffmpeg_command += os.path.abspath(filename)

        result = subprocess.check_output(ffmpeg_command, shell=True)
        print(result)

        shutil.rmtree(gif_temp_dir)

    def calculate_pop_vector_trajectory(self, dt=50 * ms, plot=False, frames_timestamps=None):
        """
        Calculates the trajectory of the center of mass over time.
        """
        denseifrs, denseisis, densetimes = self.get_dense_ifr(dt=dt, frames_timestamps=frames_timestamps)
        denseifrs3d = np.reshape(denseifrs, (denseifrs.shape[0], self.cols, self.rows))
        xsum = np.sum(denseifrs3d, axis=1)
        ysum = np.sum(denseifrs3d, axis=2)
        num_x = xsum.shape[1]
        num_y = ysum.shape[1]
        index_vector_x = np.arange(1, num_x + 1)
        index_vector_y = np.arange(1, num_y + 1)
        # calculate the weighted sum and normalize:
        x_center = np.inner(xsum, index_vector_x) / np.sum(xsum, axis=1)
        y_center = np.inner(ysum, index_vector_y) / np.sum(ysum, axis=1)
        if plot:
            plt.figure()
            plt.plot(x_center, y_center)
            plt.xlim(0, num_x)
            plt.ylim(0, num_y)
        return x_center, y_center


def interpolate_isi(ind, t=None, i=None, densetimes=None):
    """
    interplolate interspike intervals so that there is a value at all timesteps and not just at the spike times
    """
    from scipy.interpolate import interp1d
    inds = np.where(ind == i)[0]
    isitimes = t[inds]
    # if ind % 100 == 0:
    #     print(str(ind))
    if len(isitimes) > 2:
        interpf = interp1d(isitimes[1:], np.diff(
            isitimes), kind='linear', bounds_error=False, fill_value=0.0)
        return interpf(densetimes)
    else:
        return np.zeros_like(densetimes)


def visualize_3d(video):
    imv = pg.ImageView()
    imv.setImage(video)
    imv.setPredefinedGradient("thermal")
    imv.show()
    app.exec()


def export_panes(gw_paneplot, filename):
    """
    generates a figure file from a GraphicsWindow object
    :param gw_paneplot:
    :param filename:
    :return:
    """
    QtGui.QApplication.processEvents()  # without this, only the first plot is exported
    exp_img = pg.exporters.ImageExporter(gw_paneplot.scene())
    if filename is None:
        gw_paneplot.show()
    elif filename.endswith('.png'):
        exp_img.export(filename)
    elif filename.endswith('.svg'):
        exp = pg.exporters.SVGExporter(gw_paneplot.scene())
        exp.export(filename + '.svg')
    else:
        exp_img.export(filename + '_panes.png')


def create_panes(video, num_rows, slice_indices=None, colormap=cm.jet):
    """

    :param video: the 3d matrix that should be plotted
    :param slice_indices: the indices of the plotted slices in the first dim of the 3d matrix
    :param num_rows: the number of rows for the pane plot
    :return: the GraphicsWindow object
    """
    num_panes = len(slice_indices)
    gw_paneplot = pg.GraphicsWindow(title="pane plot")
    f = int(num_panes / num_rows)
    width = 1920
    gw_paneplot.resize(width, width / f * num_rows)

    if slice_indices is None:
        slice_indices = np.arange(num_panes)

    vb = dict()
    imItems = dict()
    for i in range(num_panes):
        picture = video[slice_indices[i]]
        imItems[i] = pg.ImageItem(colormap(picture / np.max(video)))
        # imItems[i].setTitle(title=str(i*timestep)) #not possible for images
        vb[i] = gw_paneplot.addViewBox()
        vb[i].addItem(imItems[i])
        if np.mod(i + 1, np.round(num_panes / num_rows)) == 0:
            gw_paneplot.nextRow()

    return gw_paneplot
