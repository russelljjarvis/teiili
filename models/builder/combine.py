

"""
This file contains a set of functions used by the library to combine models.

Combine_neu_dict is used for combining neuron models

Combine_syn_dict is used for combining neuron models

Both functions use the var_replacer when the special overwrite character '%'
is found in the equantions

"""


def combine_neu_dict(eq_templ, param_templ):
    """Function to combine neuron models into a single neuron model.
    This library offers this ability in order to combine single blocks into bigger
    and more complex Brian2 compatibile model
    combine_neu_dict also presents the possibility to delete or overwrite an explicit
    function with the use of the special character '%'.
    Example with two different dictionaries both containing the explicit function
    for the variable 'x':
        eq in the former argument: x = theta
        eq in the latter argument: %x = gamma
        eq in the output : x = gamma
    '%x' without any assignement will simply delete the variable from output
    and from the parameter dictionary.

    Args:
        *args: 2n dictionaries. The first n subset is composed of the equation
        templates selected by the builder, the second n subset is composed by
        the parameters templates for every model block i.e
        combine_neu_dict(eq_block1, eq_block2, param_dict1, param_dict2)

    Returns:
        dict: brian2-like dictionary to describe neuron model composed by
            model (string): Actually neuron model differential equation
            threshold (string): Dictionary with equations specifying behaviour of synapse to
                post-synaptic spike
            reset (string): Dictionary with equations specifying behaviour of synapse to
                pre-synaptic spike
            parameters (dict): Dictionary of parameters
    """
    model = ''
    threshold = ''
    reset = ''

    params = {}
    for d in param_templ:
        #print(d)
        params.update(d)

    for k, eq in enumerate(eq_templ):
        newmodel = eq['model']
        if '%' in newmodel:
            model, newmodel, params = var_replacer(model, newmodel, params)
        model += newmodel

        newthreshold = eq['threshold']
        if '%' in newthreshold:
            threshold, newthreshold, params = var_replacer(threshold, newthreshold, params)
        threshold += newthreshold

        newreset = eq['reset']
        if '%' in newreset:
            reset, newreset, params = var_replacer(reset, newreset, params)
        reset += newreset

    return {'model': model, 'threshold': threshold, 'reset': reset, 'parameters': params}


def combine_syn_dict(eq_templ, param_templ):

    """Function to combine synapse models into a single synapse model.
    This library offers this ability in order to combine single blocks into bigger
    and more complex Brian2 compatibile model
    combine_syn_dict also presents the possibility to delete or overwrite an explicit
    function with the use of the special character '%'.
    Example with two different dictionaries both containing the explicit function
    for the variable 'x':
        eq in the former argument: x = theta
        eq in the latter argument: %x = gamma
        eq in the output : x = gamma
    '%x' without any assignement will simply delete the variable from output
    and from the parameter dictionary.

    Args:
        *args: 2n dictionaries. The first n subset is composed of the equation
        templates selected by the builder, the second n subset is composed by
        the parameters templates for every model block i.e
        combine_syn_dict(eq_block1, eq_block2, param_dict1, param_dict2)

    Returns:
        dict: brian2-like dictionary to describe neuron model composed by
            model (string): Actually neuron model differential equation
            on_post (string): Dictionary with equations specifying behaviour of synapse to
                post-synaptic spike
            on_pre (string): Dictionary with equations specifying behaviour of synapse to
                pre-synaptic spike
            parameters (dict): Dictionary of parameters
    """
    model = ''
    on_pre = ''
    on_post = ''
    params = {}
    # eq_templ = args[0:int(len(args)/2)]
    # param_templ = args[int(len(args)/2):]

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
            on_post, newon_post, params = var_replacer(on_post, newon_post, params)
        on_post += newon_post

        params.update(param_templ[k])

    return {'model': model, 'on_pre': on_pre, 'on_post': on_post, 'parameters': params}


def var_replacer(firstEq, secondEq, params):

    """Function to delete variables from equations and parameters.
    It works with couples of strings and a dict of parameters: firstEq, secondEq and params
    It search for every line in secondEq for the special character '%' removing it,
    and then search the variable (even if in differential form '%dx/dt') and erease
    every line in fisrEq starting with that variable.(every explicit equation)
    If the character '=' or ':' is not in the line containing the variable in secondEq
    the entire line would be ereased.
    Ex:
        '%x = theta' --> 'x = theta'
        '%x' --> ''
    This feature allows to remove equations in the template that we don't want to
    compute by writing '%[variable]' in the other equation blocks.

    Args:
        firstEq (string): The first subset of equation that we want to expand or
            overwrite .
        secondEq (string): The second subset of equation wich will be added to firstEq
            It also contains '%' for overwriting or ereasing lines in
            firstEq.
        parameters (dict): The parameter dictionary of the firstEq, var_replacer
        will remove any variable deleted or replaced with the special character
        '%'

    Returns:
        resultfirstEq: The first eq string containing the replaced variable eqs
        resultsecondEq:  The second eq string without the lines containing the
            special charachter '%'
        params: The parameter dictionary not containing the removed/replaced variable
    """

    resultfirstEq = firstEq.splitlines()
    resultsecondEq = secondEq.splitlines()

    for k, line in enumerate(secondEq.splitlines()):
        if '%' in line:  # if the replace character '%' is found, extract the variable
            var = line.split('%', 1)[1].split()[0]
            line = line.replace("%", "")
            if '/' in var:
                var = var.split('/', 1)[0][1:]
            diffvar = 'd' + var + '/dt'

            for kk, line2 in enumerate(firstEq.splitlines()):
                # now look for the variable in the equation lines extracted from
                # firstEq
                if (var in line2) or (diffvar in line2):
                    # if i found a variable i need to check then if it's in explicit form
                    # meaning it's followed by ":" or "="
                    # e.g. "var = x+1"  or "var : 1"
                    if ((var == line2.replace(':', '=').split('=', 1)[0].split()[0]) or
                    (diffvar in line2.replace(':', '=').split('=', 1)[0].split()[0])):
                        resultfirstEq[kk] = line

            #after replacing the "%" flagged line in the resultfirstEq
            #remove that line from the resultsecondEq
            resultsecondEq[k] = ""
            try:
                params.pop(var)
            except KeyError:
                pass

    resultfirstEq = "\n".join(resultfirstEq)
    resultsecondEq = "\n".join(resultsecondEq)

    return resultfirstEq, resultsecondEq, params
