# copied and modified from nengo.networks.workingmemory

import nengo
import numpy as np
from nengo.networks import EnsembleArray

class GatedMemory(nengo.Network):
    def __init__(self, n_neurons, dimensions,
                 feedback=1.0, difference_gain=1.0, synapse=0.1,
                 label=None, seed=None, add_to_container=None):
        super(GatedMemory, self).__init__(label, seed, add_to_container)
        vocab = dimensions
        n_total_neurons = n_neurons * dimensions

        with self:
            # integrator to store value
            self.mem = EnsembleArray(n_neurons, dimensions, label="mem")
            nengo.Connection(self.mem.output, self.mem.input,
                             transform=feedback,
                             synapse=synapse)

            # calculate difference between stored value and input
            self.diff = EnsembleArray(n_neurons, dimensions, label="diff")
            nengo.Connection(self.mem.output, self.diff.input, transform=-1)

            # feed difference into integrator
            nengo.Connection(self.diff.output, self.mem.input,
                             transform=difference_gain,
                             synapse=synapse)

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = nengo.Node(size_in=1)
            self.diff.add_neuron_input()
            nengo.Connection(self.gate, self.diff.neuron_input,
                             transform=np.ones((n_total_neurons, 1)) * -10,
                             synapse=None)

            self.input = self.diff.input
            self.output = self.mem.output

            self.inputs = dict(default=(self.input, vocab))
            self.outputs = dict(default=(self.output, vocab))
