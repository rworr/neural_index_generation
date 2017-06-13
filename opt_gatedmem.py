# copied and modified from nengo.networks.workingmemory

import nengo
import numpy as np
import nengo_spa as spa

class GatedMemory(spa.Network):
    def __init__(self, dimensions, subdimensions=16, neurons_per_dim=100,
                 feedback=1.0, difference_gain=1.0, synapse=0.1,
                 label=None, seed=None, add_to_container=None):
        super(GatedMemory, self).__init__(label, seed, add_to_container)
        
        tau_gaba = 0.00848
        n_neurons = neurons_per_dim * dimensions
        with self:
            self.mem = spa.State(dimensions, subdimensions=subdimensions, neurons_per_dimension=neurons_per_dim, feedback=1.0)
            self.diff = spa.State(dimensions, subdimensions=subdimensions, neurons_per_dimension=neurons_per_dim)
            nengo.Connection(self.mem.output, self.diff.input, transform=-1)
            nengo.Connection(self.diff.output, self.mem.input,
                             transform=difference_gain,
                             synapse=synapse)

            self.gate = nengo.Node(size_in=1)

            self.diff.state_ensembles.add_neuron_input()
            nengo.Connection(self.gate, self.diff.state_ensembles.neuron_input,
                             transform=np.ones((n_neurons, 1)) * -10, synapse=tau_gaba)

            self.input = self.diff.input
            self.output = self.mem.output

            self.inputs = dict(default=(self.input, dimensions))
            self.outputs = dict(default=(self.output, dimensions))
