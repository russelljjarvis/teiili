try:
    from teili.tools.visualizer.DataViewers import DataViewer
except BaseException:
    from teili.teili.tools.visualizer.DataViewers import DataViewer


class RasterplotViewer(DataViewer):
    """ Parent class of Rasterplot viewers with different backends (matplotlib, pyqtgraph)"""
    pass
