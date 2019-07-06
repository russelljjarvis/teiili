#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:34:43 2017

@author: alpha

This is an idea how live plotting and parameter changing in brian2 could work using pyqtgraph
It might not work properly and there is still a lot to be done

Also be aware, that this is live, but not real time!

Attributes:
    app (TYPE): Description

"""
import sys
import numpy as np
from brian2 import asarray, Quantity
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph.ptime as ptime

from scipy import signal
from brian2 import ms, mV, pA, nS, nA

from teili import BuildingBlock

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))


class ParameterGUI(QtGui.QWidget):
    """
    Creates a GUI to change parameters while the brian2 simulation is running.
    You initialize passing the brian2 network, then, parameters are added by the add_params method.
    When all parameters are added, show_gui has to be called.

    Attributes:
        blockparams (list): Description
        net (TYPE): brian2 network
        params (list): Description
        paramtree (TYPE): Description
        state (TYPE): Description
        tree (TYPE): Description
        units (dict): Description
    """

    def __init__(self, net=None):
        """Summary

        Args:
            net (None, optional): Description
        """
        QtGui.QWidget.__init__(self)

        self.blockparams = []
        self.params = []
        self.units = {}
        # Create ParameterTree widget
        self.tree = ParameterTree()

        self.tree.setWindowTitle('Parameters')

        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(QtGui.QLabel("Set parameters here"), 0,  0, 1, 2)
        layout.addWidget(self.tree, 1, 0, 1, 1)
        #layout.addWidget(t2, 1, 1, 1, 1)
        self.setWindowTitle('Parameter settings')
        self.resize(500, 1000)

        self.params += [{'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action'}]}]

        self.net = net

#        if net is not None:
#            self.add_net(net)

    def add_params(self, parameters):
        """Summary

        Args:
            parameters (TYPE): Description

        Raises:
            Exception: Description
        """
        if self.net is None:
            raise Exception(
                'You neet to add a Net before adding single parameters')
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]

        for par in parameters:
            try:
                self.units.update({par.group_name + '__' + par.name: par.dim})
            except Exception as e:
                print(str(e))
                #units.update({par : ''})
                pass
        children = [{'name': par.group_name + '__' + par.name,
                     'type': 'float', 'suffix': ' ' + str(self.units[par.group_name + '__' + par.name]),
                     'value': np.asarray(par), 'step': abs(asarray(par) / 10)} for par in parameters]

        self.params += [{'name': 'manually added',
                         'type': 'group', 'children': children}]

    def add_net(self, net):
        """This does not work with arrays!

        Args:
            net (TYPE): Description
        """
        # This is not really working right now
        self.net = net
        for group in net:
            if hasattr(group, "standalone_params"):
                for par in group.standalone_params:
                    try:
                        self.units.update(
                            {par: group.standalone_params[par].dim})
                    except:
                        #self.units.update({par : ''})
                        pass
        for group in net.blocks:
            if hasattr(group, "standalone_params"):
                for par in group.standalone_params:
                    try:
                        self.units.update(
                            {par: group.standalone_params[par].dim})
                    except:
                        #self.units.update({par : ''})
                        pass

        # collect blockparams here in order avoid duplicates
        try:
            self.blockparams = [
                [par for par in block.standalone_params] for block in net.blocks]
            self.blockparams = [i for sl in self.blockparams for i in sl]
        except AttributeError:
            pass

        # params = [{'name': 'Input', 'type': 'group', 'children': [] for group
        # in Net if group.isinstance('BuildingBlock')]

        self.params += [{'name': group.name, 'type': 'group', 'children': self.make_children(group)}
                        for group in net if hasattr(group, "standalone_params") and len(self.make_children(group)) > 0]

        # make parameter groups for building blocks
        self.params += [{'name': block.name, 'type': 'group', 'children': self.make_children(block)}
                        for block in net.blocks]

    def show_gui(self):
        """Summary
        """
        if self.net is None:
            print('You neet to add a Net for proper function')
        else:
            # Create tree of Parameter objects
            self.paramtree = Parameter.create(
                name='params', type='group', children=self.params)

            # Too lazy for recursion:
            for child in self.paramtree.children():
                child.sigValueChanging.connect(self.gui_value_changing)
                for ch2 in child.children():
                    ch2.sigValueChanging.connect(self.gui_value_changing)

        self.paramtree.param('Save/Restore functionality',
                             'Save State').sigActivated.connect(self.save)
        self.paramtree.param('Save/Restore functionality',
                             'Restore State').sigActivated.connect(self.restore)
        self.state = self.paramtree.saveState()

        self.tree.setParameters(self.paramtree, showTop=False)

        self.show()

    def save(self):
        """Summary
        """
        self.state = self.paramtree.saveState()

    def restore(self):
        """Summary
        """
        self.paramtree.restoreState(self.state)

    def make_children(self, group):
        """Summary

        Args:
            group (TYPE): Description

        Returns:
            TYPE: Description
        """
        try:
            return [{'name': par,
                     'type': 'float',
                     'suffix': ' ' + str(self.units[par]),
                     'value': asarray(group.standalone_params[par]),
                     'step': abs(asarray(group.standalone_params[par]) / 10)}
                    for par in group.standalone_params if (par not in self.blockparams or isinstance(group, BuildingBlock))]
        except:
            return []

    def gui_value_changing(self, param, value):
        """Summary

        Args:
            param (TYPE): Description
            value (TYPE): Description
        """
        print("Value changing: %s %s" % (param, value))

        # This won't work if there is a '__' in the parameter name!
        groupname = param.name().rpartition('__')[0]
        parname = param.name().rpartition('__')[-1]

        print('groupname: ', groupname, 'parname: ', parname)

        # Quantity.with_dimensions(value,self.standalone_params[param.name()].dim)
        valWithUnit = Quantity.with_dimensions(value, self.units[param.name()])
#        except AttributeError as e:
#            print(e)
#            valWithUnit = value

        self.net[groupname].__setattr__(parname, valWithUnit)

        try:
            # this has to be written for every group that has brian2 string
            # based parameter setting
            self.net[groupname].update_param(parname)
        except AttributeError as e:
            print(e)
            pass


class PlotGUI(pg.GraphicsWindow):

    """
    This creates a GUI to plot variables while the brian2 simulation is running.
    This is for 1d variables.

    Attributes:
        data (dict): Description
        timer (TYPE): Description
    """

    def __init__(self, data=None, parent=None):
        """Summary

        Args:
            data (None, optional): Description
            parent (None, optional): Description
        """
        pg.GraphicsWindow.__init__(self, parent)

        self.resize(1000, 600)
        self.setWindowTitle('Field Plot')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

        self.curveData = {}

        if data is not None:
            self.add(data)

    def add(self, data):
        """adds another plot

        Args:
            data (TYPE): Description
        """
        p = self.addPlot(title="plot")
        self.curveData.update({p.plot(pen='y'): data})

    def update(self):
        """Summary
        """
        for curve in self.curveData:
            curve.setData(np.asarray(self.curveData[curve]))


class ImageGUI(pg.GraphicsLayoutWidget):

    """
    This creates a GUI to plot variables while the brian2 simulation is running.
    This is for 2d (image) variables.

    Attributes:
        data (TYPE): Description
        fps (int): Description
        i (int): Description
        img (TYPE): Description
        squareSize (TYPE): Description
        updateTime (TYPE): Description
    """

    def __init__(self, data=None, parent=None):
        """Summary

        Args:
            data (None, optional): Description
            parent (None, optional): Description
        """
        pg.GraphicsLayoutWidget.__init__(self, parent)

        self.data = data
        self.squareSize = int(np.sqrt(data.shape[0]))

        # Create window with GraphicsView widget
        self.show()  # show widget alone in its own window
        self.setWindowTitle('2d Field')
        view = self.addViewBox()

        # lock the aspect ratio so pixels are always square
        view.setAspectLocked(True)

        # Create image item
        self.img = pg.ImageItem(border='w')
        view.addItem(self.img)

        # Set initial view bounds
        view.setRange(QtCore.QRectF(0, 0, self.squareSize, self.squareSize))

        # Create random image
        #data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
        self.i = 0

        self.updateTime = ptime.time()
        self.fps = 0

        self.updateData()

    def updateData(self):
        """Summary
        """
        # Display the data
        self.img.setImage(np.reshape(
            self.data, (self.squareSize, self.squareSize)))
        #self.i = (self.i+1) % self.data.shape[0]

        QtCore.QTimer.singleShot(1, self.updateData)
        now = ptime.time()
        fps2 = 1.0 / (now - self.updateTime)
        self.updateTime = now
        self.fps = self.fps * 0.9 + fps2 * 0.1


#import threading
#
#t = threading.Thread(target=runLoop)
# t.start()
# sys.exit(pg.QtGui.QApplication.exec_())
#
# def runLoop():
#    while 1:
#        data = np.random.normal(size=(500,500))
#        im.setImage(data)
#        time.sleep(0.1)
