import numpy as np

from .DataModel import DataModel


class StateVariablesModel(DataModel):
    """ Model to hold data on several state variables and their measurement time points
        self.var_name ([n_neurons, n_timepoints]), t_var_name ([n_timepoints]))"""

    def __init__(
            self,
            state_variable_names=None,
            state_variables=None,
            state_variables_times=None):
        """ Setup StateVariablesModel
        Args:
            state_variable_names (list of str): list of names (str) of state variables
            state_variables (list of list/array): list of state variable values [n_timesteps, n_traces]
            state_variables_times (list of list/array): list of time points where state variables were measured
        """

        if state_variable_names is None and state_variables is None and state_variables_times is None:
            pass
        else:
            for state_var_name, state_var, state_var_times in zip(
                    state_variable_names, state_variables, state_variables_times):
                self.add_one_state_variable(
                    state_variable_name=state_var_name,
                    state_variable=state_var,
                    state_variable_times=state_var_times)

    @classmethod
    def from_brian_state_monitors(
            cls,
            brian_state_monitors,
            skip_not_rec_neuron_ids=False):
        """ Classmethod to init StateVariablesModel from brian state monitor.
            Brian allows you to decide which neurons should be recorded. If skip_not_rec_neuron_ids is True:
            self.var_name is of shape ([n_neurons_recorded, n_timepoints]), if False: self.var_name ([n_neurons_total, n_timepoints])
            whereby the rows of not recorded neurons are set to nan. (n_neurons_total = max_neuron_id +1)

        Args:
            brian_state_monitors: brian2 state monitor
            skip_not_rec_neuron_ids (bool): if True: not recorded neurons are not considered and index of recorded neurons is lost

        Remarks:
            The state variable array will be transposed from the brian state monitor, to consistenly represent time
            along axis 0 in state_variables and in state_variables_times

            """

        state_variable_names, state_variables, state_variables_times = [], [], []

        for brian_state_mon in brian_state_monitors:
            num_timesteps = len(brian_state_mon.t)
            num_neurons_recorded = len(np.asarray(brian_state_mon.record))
            max_neuron_ids_recorded = np.max(
                np.asarray(brian_state_mon.record))

            for state_var_name in brian_state_mon.recorded_variables:

                if skip_not_rec_neuron_ids:
                    state_var_array = np.empty(
                        [num_timesteps, num_neurons_recorded])
                    state_var_array.fill(np.nan)
                    neuron_nrs = range(num_neurons_recorded)
                else:
                    state_var_array = np.empty(
                        [num_timesteps, max_neuron_ids_recorded + 1])
                    state_var_array.fill(np.nan)
                    neuron_nrs = brian_state_mon.record

                for state_var_per_neuron, neuron_nr in zip(
                        getattr(brian_state_mon, state_var_name), neuron_nrs):
                    state_var_array[:, neuron_nr] = np.asarray(
                        state_var_per_neuron)

                state_variable_names.append(state_var_name)
                state_variables.append(state_var_array)
                state_variables_times.append(np.asarray(brian_state_mon.t))

        newStateVariableModel = cls(
            state_variable_names=state_variable_names,
            state_variables=state_variables,
            state_variables_times=state_variables_times)

        return newStateVariableModel

    def add_one_state_variable(
            self,
            state_variable_name,
            state_variable,
            state_variable_times):
        """ Add one state variable and its measurement time points to StateVariablesModel
        Args:
            state_variable_name (str): name (str) of state variable
            state_variable (list/array): state variable values
            state_variable_times (list/array): time points where state variable was measured
        """
        setattr(self, state_variable_name, state_variable)
        setattr(self, 't_' + state_variable_name, state_variable_times)
