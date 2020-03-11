#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:43:12 2018

@author: alpha
"""
import sys
from brian2 import us, ms
from teili.tools.plotter2d import Plotter2d
from teili.tools.converter import aedat2numpy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))


import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

print('Please select a file...\n[Can be either .npy or .aedat.]')
eventsfile_Cam1 = filedialog.askopenfilename()
print('Please select a file...\n[Can be either .npy or .aedat.]')
eventsfile_Cam2 = filedialog.askopenfilename()
if eventsfile_Cam1[-5:] == 'aedat':
    eventsfile_Cam1 = aedat2numpy(eventsfile_Cam1, camera='DAVIS240')
if eventsfile_Cam2[-5:] == 'aedat':
    eventsfile_Cam2 = aedat2numpy(eventsfile_Cam2, camera='DAVIS240')

spmon2d_Cam1 = Plotter2d.loaddvs(eventsfile_Cam1)
spmon2d_Cam2 = Plotter2d.loaddvs(eventsfile_Cam2)
imv1 = spmon2d_Cam1.plot3d_on_off(plot_dt=100 * ms, filtersize=250 * ms)
imv2 = spmon2d_Cam2.plot3d_on_off(plot_dt=100 * ms, filtersize=250 * ms)

imv3 = spmon2d_Cam1.plot3d(plot_dt=100 * ms, filtersize=250 * ms)
imv4 = spmon2d_Cam2.plot3d(plot_dt=100 * ms, filtersize=250 * ms)
# imv1.show()

# Create window with ImageView widget
win = QtGui.QDialog()
#gridlayout =  QtGui.QHBoxLayout(win)
#gridlayout =  QtGui.QVBoxLayout(win)
gridlayout = QtGui.QGridLayout(win)
gridlayout.addWidget(imv1, 1, 1)
gridlayout.addWidget(imv2, 2, 1)
gridlayout.addWidget(imv3, 1, 2)
gridlayout.addWidget(imv4, 2, 2)
win.resize(1500, 1000)
win.setLayout(gridlayout)
win.show()
win.setWindowTitle('DVS plot')
imv1.play(50)
imv2.play(50)
imv3.play(50)
imv4.play(50)

app.exec()
#spmon2d_Cam1.plot_panes(num_panes=40, filtersize=100 * ms, num_rows=4, filename=None)

# please provide an absolute path, otherwise dvs.gif will be stored in your wd!
#spmon2d_Cam1.generate_gif('dvs2.gif', plotfunction = 'plot3d_on_off', filtersize=100 * ms, plot_dt=50 * ms)
