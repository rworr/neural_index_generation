import nengo
import numpy as np
import nengo_spa as spa
import matplotlib.pyplot as plt

from nengo.networks import InputGatedMemory
from nengo_spa.pointer import SemanticPointer
from nengo_spa.pointer import AbsorbingElement

seed = np.random.randint(100)
np.random.seed(seed)
print seed

dim = 64
isi = 2.0
neurons_per_dim = 200
subdimensions = 1
cconv_neurons = 200
sim_time = 120.0

vocab = spa.Vocabulary(dim)
vocab.add('POS', SemanticPointer(dim).unitary())

def clock(t):
    if (t % isi) < isi/2.0:
        return 1
    return 0

def init_pos(t):
    if t < isi:
        return 'POS'
    return '0'

with spa.Network('IG', vocabs=[vocab], seed=seed) as model:
    model.clock = nengo.Node(clock)
    model.inv_clock = nengo.Node(size_in=1)
    nengo.Connection(model.clock, model.inv_clock, transform=-1)
    nengo.Connection(nengo.Node(1), model.inv_clock)

    model.pos_ptr = nengo.Node(output=vocab['POS'].v)
    model.position = spa.State(vocab)
    model.inp = spa.Input()
    model.inp.position=init_pos
    
    model.current = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    model.next = InputGatedMemory(n_neurons=neurons_per_dim, dimensions=dim)
    model.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    nengo.Connection(model.inv_clock, model.current.gate)
    nengo.Connection(model.clock, model.next.gate)

    nengo.Connection(model.position.output, model.current.input)
    nengo.Connection(model.current.output, model.cconv.input_a)
    nengo.Connection(model.pos_ptr, model.cconv.input_b)
    nengo.Connection(model.cconv.output, model.next.input)
    nengo.Connection(model.next.output, model.current.input)

    cur_probe = nengo.Probe(model.current.output, synapse=0.03)
    next_probe = nengo.Probe(model.next.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(sim_time)
t = sim.trange()

cur_data = sim.data[cur_probe]
next_data = sim.data[next_probe]

def plot_data(data, title):
    ymin, ymax = -0.2, 1.5
    plt.figure()
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel('Time')
    plt.xlim(right=t[-1])
    plt.legend(vocab.keys(), loc='upper center')

def plot_vector_length(data, title):
    data = np.asarray([np.linalg.norm(x) for x in data])
    plot_data(data, title)

def plot_index_sim(cur, next):
    data = np.asarray([np.dot(cur[i], next[i]) for i in range(0, len(cur))])
    plot_data(data, 'dot(CUR, NEXT)')

def plot_absorbing_sim(cur):
    abs_ptr = AbsorbingElement(dim)
    data = np.asarray([np.dot(cur[i], abs_ptr.v) for i in range(0, len(cur))])
    plot_data(data, 'dot(CUR, ABS)')

plot_index_sim(cur_data, next_data)
plot_absorbing_sim(cur_data)
plot_vector_length(cur_data, 'Current Pointer Length')
plt.show()
