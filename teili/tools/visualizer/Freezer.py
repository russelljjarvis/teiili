class Freezer(object):
    """ class to freeze a class so that no new attributes can be added after self._freeze() is called as it raises
     a TypeError. This is helpful to ensure that the correct attribute names are used (e.g. avoid creating the same
     attribute twice but spelled differently)."""
    
    __isfrozen = False

    def __setattr__(self, key, value):
        """ Function which checks if "key" is already an attribute of self and if not raises a TypeError
        if self._isfrozen set to True.

        Args:
            key (str): name of attribute to set
            value: value of attribute to set
        """

        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        """ Function to set self._isfrozen to True. """
        self.__isfrozen = True
