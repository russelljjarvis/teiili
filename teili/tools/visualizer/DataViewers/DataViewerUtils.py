from abc import ABC, abstractmethod

class DataViewerUtils(ABC):
    """ Parent class to classes which hold backend specific methods which
        are shared between different Viewers of the same backend"""
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