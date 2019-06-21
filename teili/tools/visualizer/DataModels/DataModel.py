# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich
import numpy as np

class DataModel(object):
    """ Parent class of all DataModels """

    def __init__(self):
        pass

    @classmethod
    def from_file(cls, path_to_file):
        """ Classmethod to initialize DataModel object from DataModel object
            stored in npz-file at path_to_file """
        newDataModel = cls()
        newDataModel.load_datamodel(path_to_file)
        return newDataModel

    def save_datamodel(self, outputfilename):
        """ Save DataModel object to outputfilename with npz"""
        arrays_to_save_by_name = {}
        for attr_name in self.attributes_to_save:
            arrays_to_save_by_name[attr_name] = getattr(self, attr_name)
        np.savez(outputfilename, **arrays_to_save_by_name)

    def load_datamodel(self, path_to_data):
        """ Load DataModel instance from path_to_data """
        data = np.load(path_to_data)
        for varible_name, variable_values in data.items():
            setattr(self, varible_name, variable_values)
        data.close()
