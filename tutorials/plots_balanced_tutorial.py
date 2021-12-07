import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from pathlib import Path
import pickle

batch_mean, batch_height = {'4': [], '8': []}, {'4': [], '8': []}
for file_desc in sorted(Path('./').glob('res*')):
    res = str(file_desc)[3]
    with open(file_desc, 'rb') as f:
        trials = pickle.load(f)
    batch_mean[res].append(np.mean(trials))
    batch_height[res].append(np.std(trials) / np.sqrt(len(trials)))

win = pg.GraphicsLayoutWidget(show=True)
res = ['4', '8']
wi_perc = np.linspace(.1, 1, 10)
rows = 2
for r in res:
    batch_mean[r] = np.array(batch_mean[r])
    batch_height[r] = np.array(batch_height[r])

    plt = win.addPlot(title=f'{r}-bit resolution')
    err = pg.ErrorBarItem(x=wi_perc, y=batch_mean[r], height=batch_height[r], beam=.02)
    plt.addItem(err)
    plt.plot(wi_perc, batch_mean[r], symbol='o', pen={'color': 0.8, 'width': 2})
    plt.setLabel('left', 'Rate', units='Hz')
    plt.setLabel('bottom', 'inhibitory weight strength (%)')
    rows -= 1
    if rows:
        win.nextRow()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
