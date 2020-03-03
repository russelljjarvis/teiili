import numpy as np

from brian2 import us, ms, second, prefs, defaultclock, start_scope, SpikeGeneratorGroup, SpikeMonitor, StateMonitor

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataViewers import PlotSettings


def get_plotsettings(alpha=None, max255=False):

    if alpha is not None:
        colors = np.asarray([(1,0,0, alpha),  # 'r',
                             (0,1,0, alpha),  # 'g',
                             (0,0,1, alpha)])  # 'b'])
        if max255:
            colors = colors * 255.

    else:
        colors = ['r', 'g', 'b']

    MyPlotSettings = PlotSettings(
        fontsize_title=20,
        fontsize_legend=14,
        fontsize_axis_labels=14,
        marker_size=30,
        colors=colors)
    return MyPlotSettings

def run_brian_network(spikemonitors=True, statemonitors=True):
    prefs.codegen.target = "numpy"
    defaultclock.dt = 10 * us

    start_scope()
    N_input, N_N1, N_N2 = 1, 5, 3
    duration_sim = 30  # ms

    Net = TeiliNetwork()

    # setup spike generator
    spikegen_spike_times = np.sort(np.random.choice(size=500, a=np.arange(float(defaultclock.dt), float(duration_sim*ms)*0.9,
                                                                          float(defaultclock.dt*5)), replace=False)) * second

    spikegen_neuron_ids = np.zeros_like(spikegen_spike_times) / ms
    gInpGroup = SpikeGeneratorGroup(
        N_input,
        indices=spikegen_neuron_ids,
        times=spikegen_spike_times,
        name='gtestInp')

    # setup neurons
    testNeurons1 = Neurons(
        N_N1, equation_builder=DPI(
            num_inputs=2), name="testNeuron")
    testNeurons1.set_params(neuron_model_param)
    testNeurons2 = Neurons(
        N_N2, equation_builder=DPI(
            num_inputs=2), name="testNeuron2")
    testNeurons2.set_params(neuron_model_param)

    # setup connections
    InpSyn = Connections(
        gInpGroup,
        testNeurons1,
        equation_builder=DPISyn(),
        name="testSyn",
        verbose=False)
    InpSyn.connect(True)
    InpSyn.weight = '200 + rand() * 100'
    Syn = Connections(
        testNeurons1,
        testNeurons2,
        equation_builder=DPISyn(),
        name="testSyn2")
    Syn.connect(True)
    Syn.weight = '200 + rand() * 100'

    Net.add(
        gInpGroup,
        testNeurons1,
        testNeurons2,
        InpSyn,
        Syn)

    returns = []
    if spikemonitors:
        # spike monitors input and network
        spikemonN1 = SpikeMonitor(testNeurons1, name='spikemon')
        spikemonN2 = SpikeMonitor(testNeurons2, name='spikemonOut')
        Net.add(spikemonN1, spikemonN2)
        returns.extend([spikemonN1, spikemonN2])

    if statemonitors:
        # state monitor neurons
        statemonN1 = StateMonitor(testNeurons1, variables=["Iin", "Iahp"], record=True, name='statemonNeu')
        statemonN2 = StateMonitor(testNeurons2, variables=['Imem'], record=True, name='statemonNeuOut')
        Net.add(statemonN1, statemonN2)
        returns.extend([statemonN1, statemonN2])

    Net.run(duration_sim * ms)
    return returns