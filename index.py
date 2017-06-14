# Complete index generation tests: regular, nengo_spa (with old InputGatedMemory), and nengo_spa with fixed memory

import os
import nengo
import nengo_spa
import numpy as np
import matplotlib.pyplot as plt

from nengo import spa
from nengo.networks import InputGatedMemory
from nengo_spa.pointer import SemanticPointer
from nengo_spa.pointer import AbsorbingElement
from opt_gatedmem import GatedMemory

seed = np.random.randint(100)
np.random.seed(seed)
print 'Seed:', seed

dim = 64
isi = 2.0
neurons_per_dim = 200
subdimensions = 1
cconv_neurons = 200
sim_time = 120.0

vocab = spa.Vocabulary(dim)
vocab.add('POS', vocab.create_pointer(unitary=True))

path = os.path.join('.', str(seed))
if not os.path.exists(path):
    os.makedirs(path)

## Helper functions
# Plot a graph
def plot_data(data, title):
    ymin, ymax = -1.7, 1.7
    plt.figure()
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel('Time')
    plt.xlim(right=t[-1])
    figpath = os.path.join(path, title.replace(' ', '_') + str(seed) + '.png')
    plt.savefig(figpath)

# Plot vector length
def plot_vector_length(data, title):
    data = np.asarray([np.linalg.norm(x) for x in data])
    plot_data(data, title)

# Plot unnormalized similarity between two probes
def plot_index_sim(cur, next, title):
    data = np.asarray([np.dot(cur[i], next[i]) for i in range(0, len(cur))])
    plot_data(data, title + ' dot(CUR, NEXT)')

# Plot similarity to the absorbing element
def plot_absorbing_sim(cur, title):
    abs_ptr = AbsorbingElement(dim)
    data = np.asarray([np.dot(cur[i], abs_ptr.v) for i in range(0, len(cur))])
    plot_data(data, title + ' dot(CUR, ABS)')

# Generate all graphs
def generate_graphs(cur_data, next_data, title_prefix):
    plot_index_sim(cur_data, next_data, title_prefix)
    plot_absorbing_sim(cur_data, title_prefix)
    plot_vector_length(cur_data, title_prefix + ' Current Pointer Length')

def clock(t):
    if (t % isi) < isi/2.0:
        return 1
    return 0

def init_pos(t):
    if t < isi:
        return 'POS'
    return '0'

# Regular index generation circuit
# Basically just an externally-clocked flip-flop
with spa.SPA('RegularIG', vocabs=[vocab], seed=seed) as rmodel:
    # External clock
    rmodel.clock = nengo.Node(clock)
    # Inverted clock
    rmodel.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(rmodel.clock, rmodel.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), rmodel.inv_clock)

    # POS representation node for convolution
    rmodel.pos_ptr = nengo.Node(output=vocab['POS'].v)

    # POS input to kick-start the process
    rmodel.position = spa.State(dim)
    rmodel.inp = spa.Input()
    rmodel.inp.position=init_pos
    
    # Gated memories for the current position and next position
    rmodel.current = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    rmodel.next = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    rmodel.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    # Clocks connected to gates (0 = set to input, 1 = hold memory)
    nengo.Connection(rmodel.inv_clock, rmodel.current.gate)
    nengo.Connection(rmodel.clock, rmodel.next.gate)

    # input -> current
    # next -> current
    # current * POS -> next
    nengo.Connection(rmodel.position.output, rmodel.current.input)
    nengo.Connection(rmodel.current.output, rmodel.cconv.input_a)
    nengo.Connection(rmodel.pos_ptr, rmodel.cconv.input_b)
    nengo.Connection(rmodel.cconv.output, rmodel.next.input)
    nengo.Connection(rmodel.next.output, rmodel.current.input)

    cur_probe = nengo.Probe(rmodel.current.output, synapse=0.03)
    next_probe = nengo.Probe(rmodel.next.output, synapse=0.03)

with nengo.Simulator(rmodel) as rsim:
    rsim.run(sim_time)
t = rsim.trange()
generate_graphs(rsim.data[cur_probe], rsim.data[next_probe], "Regular")

# New vocab under nengo_spa
vocab = nengo_spa.Vocabulary(dim)
vocab.add('POS', SemanticPointer(dim).unitary())

# Index generation under nengo_spa
with nengo_spa.Network('NengoSpaIG', vocabs=[vocab], seed=seed) as smodel:
    smodel.clock = nengo.Node(clock)
    smodel.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(smodel.clock, smodel.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), smodel.inv_clock)

    smodel.pos_ptr = nengo.Node(output=vocab['POS'].v)
    smodel.position = nengo_spa.State(vocab)
    smodel.inp = nengo_spa.Input()
    smodel.inp.position=init_pos
    
    smodel.current = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    smodel.next = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    smodel.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    nengo.Connection(smodel.inv_clock, smodel.current.gate)
    nengo.Connection(smodel.clock, smodel.next.gate)

    nengo.Connection(smodel.position.output, smodel.current.input)
    nengo.Connection(smodel.current.output, smodel.cconv.input_a)
    nengo.Connection(smodel.pos_ptr, smodel.cconv.input_b)
    nengo.Connection(smodel.cconv.output, smodel.next.input)
    nengo.Connection(smodel.next.output, smodel.current.input)

    cur_probe = nengo.Probe(smodel.current.output, synapse=0.03)
    next_probe = nengo.Probe(smodel.next.output, synapse=0.03)

with nengo.Simulator(smodel) as ssim:
    ssim.run(sim_time)
t = ssim.trange()
generate_graphs(ssim.data[cur_probe], ssim.data[next_probe], "NengoSpa")

# Optimal index generation with nengo_spa.state in the gated memory
with nengo_spa.Network('OptIG', vocabs=[vocab], seed=seed) as omodel:
    omodel.clock = nengo.Node(clock)
    omodel.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(omodel.clock, omodel.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), omodel.inv_clock)

    omodel.pos_ptr = nengo.Node(output=vocab['POS'].v)
    omodel.position = nengo_spa.State(vocab)
    omodel.inp = nengo_spa.Input()
    omodel.inp.position=init_pos
    
    omodel.current = GatedMemory(dim, subdimensions=subdimensions, neurons_per_dim=neurons_per_dim)
    omodel.next = GatedMemory(dim, subdimensions=subdimensions, neurons_per_dim=neurons_per_dim)
    omodel.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    nengo.Connection(omodel.inv_clock, omodel.current.gate)
    nengo.Connection(omodel.clock, omodel.next.gate)

    nengo.Connection(omodel.position.output, omodel.current.input)
    nengo.Connection(omodel.current.output, omodel.cconv.input_a)
    nengo.Connection(omodel.pos_ptr, omodel.cconv.input_b)
    nengo.Connection(omodel.cconv.output, omodel.next.input)
    nengo.Connection(omodel.next.output, omodel.current.input)

    cur_probe = nengo.Probe(omodel.current.output, synapse=0.03)
    next_probe = nengo.Probe(omodel.next.output, synapse=0.03)

with nengo.Simulator(omodel) as osim:
    osim.run(sim_time)
t = osim.trange()
generate_graphs(osim.data[cur_probe], osim.data[next_probe], "OptSpa")
