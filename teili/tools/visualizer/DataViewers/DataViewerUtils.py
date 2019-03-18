from abc import ABC, abstractmethod

class DataViewerUtils(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def show(self, **kwargs):
        """ Method to show plot"""
        pass

    @abstractmethod
    def save(self, **kwargs):
        """ Method to save plot"""
        pass