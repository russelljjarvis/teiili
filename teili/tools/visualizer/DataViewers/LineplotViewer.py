try:
    from teili.tools.visualizer.DataViewers import DataViewer
except BaseException:
    from teili.teili.tools.visualizer.DataViewers import DataViewer


class LineplotViewer(DataViewer):
    """ Parent class of Lineplot viewers with different backends (matplotlib, pyqtgraph)"""
    pass
