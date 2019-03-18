import _pickle as pickle

class DataModel(object):
    """ Parent class of all DataModels """

    def __init__(self):
        pass

    @classmethod
    def from_file(cls, path_to_file):
        """ Classmethod to initialize DataModel object from DataModel object
            stored in pickle-file at path_to_file """
        newDataModel = cls()
        newDataModel.load_datamodel(path_to_file)
        return newDataModel

    def save_datamodel(self, outputfilename):
        """ Save DataModel object to outputfilename with pickle"""
        with open(outputfilename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load_datamodel(self, path_to_data):
        """ Load DataModel instance from path_to_data """
        with open(path_to_data + '.pkl', 'rb') as f:
            data = pickle.load(f)

        for varible_name, variable_values in vars(data).items():
            setattr(self, varible_name, variable_values)
