# All index generation tests, run using the same seed
# 1. Ideal index vectors (passed in through external function)
# 2. Index generation circuit built using nengo and spa
# 3. Index generation circuit built using nengo_spa, with InputGatedMemory
# 4. Index generation circuit built using nengo_spa, with nengo_spa.State in the gated memory

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

# Create new directory for the run
path = os.path.join('.', str(seed))
if not os.path.exists(path):
    os.makedirs(path)

# Create unitary vector, and pass it to vocabs for nengo and nengo_spa
ptr = SemanticPointer(dim).unitary()
nvocab = spa.Vocabulary(dim)
nvocab.add('POS', nvocab.create_pointer(unitary=True))
svocab = nengo_spa.Vocabulary(dim)
svocab.add('POS', ptr)

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
    figpath = os.path.join(path, title + '_' + str(seed) + '.png')
    plt.savefig(figpath)

# Plot vector length
def plot_vector_length(data, title):
    data = np.asarray([np.linalg.norm(x) for x in data])
    plot_data(data, title)

# Normalized similarity
def normdot(cur, next):
    curn = np.linalg.norm(cur)
    if curn != 0:
        cur = cur/curn
    nextn = np.linalg.norm(next)
    if nextn != 0:
        next = next/nextn
    return np.dot(cur, next)

# Plot normalized similarity between two probes
def plot_index_sim_normalized(cur, next, title):
    data = np.asarray([normdot(cur[i], next[i]) for i in range(0, len(cur))])
    plot_data(data, title + '_norm_dot_CUR_NEXT')

# Plot unnormalized similarity between two probes
def plot_index_sim(cur, next, title):
    data = np.asarray([np.dot(cur[i], next[i]) for i in range(0, len(cur))])
    plot_data(data, title + '_dot_CUR_NEXT')

# Plot similarity to the absorbing element
def plot_absorbing_sim(cur, title):
    abs_ptr = AbsorbingElement(dim)
    data = np.asarray([np.dot(cur[i], abs_ptr.v) for i in range(0, len(cur))])
    plot_data(data, title + '_dot_CUR_ABS')

# Generate all graphs
def generate_graphs(cur_data, next_data, title_prefix):
    plot_index_sim_normalized(cur_data, next_data, title_prefix)
    plot_index_sim(cur_data, next_data, title_prefix)
    plot_absorbing_sim(cur_data, title_prefix)
    plot_vector_length(cur_data, title_prefix + '_CUR_length')

def clock(t):
    if (t % isi) < isi/2.0:
        return 1
    return 0

def init_pos(t):
    if t < isi:
        return 'POS'
    return '0'

# 1. Ideal index vectors, passed in from an external function
# Function to pass in the current index
def index_current(t):
    cur = 'POS'
    for i in range(0, int(t // isi)):
        cur += ' * POS'
    return cur

# Function to pass in the next index
def index_next(t):
    if t < isi // 2.0:
        return '0'

    next = 'POS * POS'
    # shift time to mimic the gated memory circuit
    st = t - isi/2.0
    for i in range(0, int(st // isi)):
        next += ' * POS'
    return next

# Model is just the states representing the vectors
with spa.SPA('IdealIG', vocabs=[nvocab], seed=seed) as imodel:
    imodel.current = spa.State(dim, subdimensions=subdimensions, neurons_per_dimension=neurons_per_dim)
    imodel.next = spa.State(dim, subdimensions=subdimensions, neurons_per_dimension=neurons_per_dim)
    imodel.input = spa.Input(current=index_current, next=index_next)

    cur_probe = nengo.Probe(imodel.current.output, synapse=0.03)
    next_probe = nengo.Probe(imodel.next.output, synapse=0.03)

with nengo.Simulator(imodel) as isim:
    isim.run(sim_time)
t = isim.trange()
generate_graphs(isim.data[cur_probe], isim.data[next_probe], "Ideal")

# 2. Nengo and spa implementation
with spa.SPA('NengoIG', vocabs=[nvocab], seed=seed) as nmodel:
    # External clock
    nmodel.clock = nengo.Node(clock)
    # Inverted clock
    nmodel.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(nmodel.clock, nmodel.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), nmodel.inv_clock)

    # POS representation node for convolution
    nmodel.pos_ptr = nengo.Node(output=nvocab['POS'].v)

    # POS input to kick-start the process
    nmodel.position = spa.State(dim)
    nmodel.inp = spa.Input()
    nmodel.inp.position=init_pos
    
    # Gated memories for the current position and next position
    nmodel.current = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    nmodel.next = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)

    # Convolution network
    nmodel.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    # Clocks connected to gates (0 = set to input, 1 = hold memory)
    nengo.Connection(nmodel.inv_clock, nmodel.current.gate)
    nengo.Connection(nmodel.clock, nmodel.next.gate)

    # Connect everything together
    nengo.Connection(nmodel.position.output, nmodel.current.input)
    nengo.Connection(nmodel.current.output, nmodel.cconv.input_a)
    nengo.Connection(nmodel.pos_ptr, nmodel.cconv.input_b)
    nengo.Connection(nmodel.cconv.output, nmodel.next.input)
    nengo.Connection(nmodel.next.output, nmodel.current.input)

    cur_probe = nengo.Probe(nmodel.current.output, synapse=0.03)
    next_probe = nengo.Probe(nmodel.next.output, synapse=0.03)

with nengo.Simulator(nmodel) as nsim:
    nsim.run(sim_time)
t = nsim.trange()
generate_graphs(nsim.data[cur_probe], nsim.data[next_probe], "Nengo")

# 3. Index generation with nengo_spa, using same InputGatedMemory
with nengo_spa.Network('NengoSpaIG', vocabs=[svocab], seed=seed) as smodel:
    # External clock
    smodel.clock = nengo.Node(clock)
    # Inverted clock
    smodel.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(smodel.clock, smodel.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), smodel.inv_clock)

    # POS representation for convolution
    smodel.pos_ptr = nengo.Node(output=svocab['POS'].v)

    # Input to kick-start the process
    smodel.position = nengo_spa.State(svocab)
    smodel.inp = nengo_spa.Input()
    smodel.inp.position=init_pos
    
    # Gated memories for current and next
    smodel.current = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    smodel.next = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)

    # Convolution network
    smodel.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    # Connect the clocks to the memory gates
    nengo.Connection(smodel.inv_clock, smodel.current.gate)
    nengo.Connection(smodel.clock, smodel.next.gate)

    # Connect everything together
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

# 3. Optimal index generation with nengo_spa.state in the gated memory
with nengo_spa.Network('OptIG', vocabs=[svocab], seed=seed) as omodel:
    # External clock
    omodel.clock = nengo.Node(clock)
    # Inverted clock
    omodel.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(omodel.clock, omodel.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), omodel.inv_clock)

    # POS node for convolution
    omodel.pos_ptr = nengo.Node(output=svocab['POS'].v)

    # Input to kick-start the process
    omodel.position = nengo_spa.State(svocab)
    omodel.inp = nengo_spa.Input()
    omodel.inp.position=init_pos
    
    # Gated memories (built with nengo_spa.state) for current and next
    omodel.current = GatedMemory(dim, subdimensions=subdimensions, neurons_per_dim=neurons_per_dim)
    omodel.next = GatedMemory(dim, subdimensions=subdimensions, neurons_per_dim=neurons_per_dim)

    # Convolution network
    omodel.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    # Connect clocks to the gates
    nengo.Connection(omodel.inv_clock, omodel.current.gate)
    nengo.Connection(omodel.clock, omodel.next.gate)

    # Connect everything else
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
generate_graphs(osim.data[cur_probe], osim.data[next_probe], "Opt")
