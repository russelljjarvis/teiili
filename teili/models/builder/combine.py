#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file contains a set of functions used by the library to combine models.

Combine_neu_dict is used for combining neuron models
Combine_syn_dict is used for combining neuron models

Both functions use the var_replacer when the special overwrite character '%'
is found in the equations

Example:
    To use combine_neu_dict:

    >>> from teili.models.builder.combine import combine_neu_dict
    >>> combine_neu_dict(eq_templ, param_templ)

    To use combine_syn_dict:
    >>> from teili.models.builder.combine import combine_syn_dict
    >>> combine_syn_dict(eq_templ, param_templ)
"""


def combine_neu_dict(eq_templ, param_templ):
    """Function to combine neuron models into a single neuron model.

    This library offers this ability in order to combine single blocks into bigger
    and more complex Brian2 compatible models.
    combine_neu_dict also makes it possible to delete or overwrite an explicit
    function with the use of the special character '%'.
    Example:
        Example with two different dictionaries both containing the explicit function
        for the variable 'x':

        >>> x = theta
        >>> %x = gamma
        >>> x = gamma

        '%x' without any assignment will simply delete the variable from the output
        and from the parameter dictionary.

        To combine two dictionaries:

        >>> combine_neu_dict(eq_templ, param_templ)
    Args:
        eq_templ (dict): Dictionary containing different keywords and equations
        param_templ (dict): Dictionary containing different parameters for the
            equation

    Returns:
        dict: ['model'] Actually neuron model differential equation.
        dict: ['threshold'] Dictionary with equations specifying behaviour of synapse to post-synaptic spike.
        dict: ['reset'] Dictionary with equations specifying behaviour of synapse to pre-synaptic spike.
        dict: ['parameters'] Dictionary of parameters.
    """
    model = ''
    threshold = ''
    reset = ''

    params = {}
    for d in param_templ:
        params.update(d)

    for k, eq in enumerate(eq_templ):
        newmodel = eq['model']
        if '%' in newmodel:
            model, newmodel, params = var_replacer(model, newmodel, params)
        model += newmodel

        newthreshold = eq['threshold']
        if '%' in newthreshold:
            threshold, newthreshold, params = var_replacer(
                threshold, newthreshold, params)
        threshold += newthreshold

        newreset = eq['reset']
        if '%' in newreset:
            reset, newreset, params = var_replacer(reset, newreset, params)
        reset += newreset

    return {'model': model, 'threshold': threshold, 'reset': reset, 'parameters': params}


def combine_syn_dict(eq_templ, param_templ):
    """Function to combine synapse models into a single synapse model.

    This library offers this ability in order to combine single blocks into bigger
    and more complex Brian2 compatible models.
    combine_syn_dict also makes it possible to delete or overwrite an explicit
    function with the use of the special character '%'.
    Example with two different dictionaries both containing the explicit function
    for the variable 'x':
    Example:
        Example with two different dictionaries both containing the explicit function
        for the variable 'x':

        >>> x = theta
        >>> %x = gamma
        >>> x = gamma

        '%x' without any assignment will simply delete the variable from the output
        and from the parameter dictionary.

        To combine two dictionaries:

        >>> combine_syn_dict(eq_templ, param_templ)

    Args:
        eq_templ (dict): Dictionary containing different keywords and equations.
        param_templ (dict): Dictionary containing different parameters for the
            equation.

    Returns:
        dict: ['model'] Actually neuron model differential equation.
        dict: ['on_post'] Dictionary with equations specifying behaviour of synapse to post-synaptic spike.
        dict: ['on_pre'] Dictionary with equations specifying behaviour of synapse to pre-synaptic spike.
        dict: ['parameters'] Dictionary of parameters.
    """
    model = ''
    on_pre = ''
    on_post = ''
    params = {}

    for k, eq in enumerate(eq_templ):

        newmodel = eq['model']
        if '%' in eq['model']:
            model, newmodel, params = var_replacer(model, newmodel, params)
        model += newmodel

        newon_pre = eq['on_pre']
        if '%' in eq['on_pre']:
            on_pre, newon_pre, params = var_replacer(on_pre, newon_pre, params)
        on_pre += newon_pre

        newon_post = eq['on_post']
        if '%' in eq['on_post']:
            on_post, newon_post, params = var_replacer(
                on_post, newon_post, params)
        on_post += newon_post

        params.update(param_templ[k])

    return {'model': model, 'on_pre': on_pre, 'on_post': on_post, 'parameters': params}


def var_replacer(first_eq, second_eq, params):
    """Function to delete variables from equations and parameters.

    It works with couples of strings and a dict of parameters: first_eq, second_eq and params
    It searches for every line in second_eq for the special character '%' removing it,
    and then searching for the variable (even if in differential form '%dx/dt') and erasing
    every line in first_eq starting with that variable (every explicit equation).
    If the character '=' or ':' is not in the line containing the variable in second_eq
    the entire line would be erased.

    Args:
        first_eq (string): The first subset of equations that we want to expand or
            overwrite.
        second_eq (string): The second subset of equations which will be added to first_eq.
            It also contains '%' for overwriting or erasing lines in first_eq.
        params (dict): Dictionary of parameters to be replaced.

    Returns:
        result_first_eq: The first_eq string containing the replaced variable equations.
        result_second_eq: The second_eq string without the lines containing the special character '%'.
        params: The parameter dictionary not containing the removed/replaced variables.

    Examples:

        >>> '%x = theta' --> 'x = theta'
        >>> '%x' --> ''

        This feature allows to remove equations in the template that we don't want to
        compute by writing '%[variable]' in the other equation blocks.

        To replace variables and lines:

        >>> from teili.models.builder.combine import var_replacer
        >>> var_replacer(first_eq, second_eq, params)
    """

    result_first_eq = first_eq.splitlines()
    result_second_eq = second_eq.splitlines()

    for k, line in enumerate(second_eq.splitlines()):
        if '%' in line:  # if the replace character '%' is found, extract the variable
            var = line.split('%', 1)[1].split()[0]
            line = line.replace("%", "")
            if '/' in var:
                var = var.split('/', 1)[0][1:]
                remove_flag = False
            else:
                remove_flag = True

            diffvar = 'd' + var + '/dt'

            for kk, line2 in enumerate(first_eq.splitlines()):
                # now look for the variable in the equation lines extracted from
                # first_eq
                if (var in line2) or (diffvar in line2):
                    # if i found a variable i need to check then if it's in explicit form
                    # meaning it's followed by ":" or "="
                    # e.g. "var = x+1"  or "var : 1"
                    if ((var == line2.replace(':', '=').split('=', 1)[0].split()[0]) or
                            (diffvar in line2.replace(':', '=').split('=', 1)[0].split()[0])):
                        result_first_eq[kk] = line

            # after replacing the "%" flagged line in the result_first_eq
            # remove that line from the result_second_eq
            result_second_eq[k] = ""

            try:
                if remove_flag:
                    params.pop(var)
                else:
                    pass
            except KeyError:
                pass

    result_first_eq = "\n".join(result_first_eq)
    result_second_eq = "\n".join(result_second_eq)

    return result_first_eq, result_second_eq, params
