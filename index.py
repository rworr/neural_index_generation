import nengo
import numpy as np
import nengo_spa as spa

from nengo_spa.pointer import SemanticPointer
import matplotlib.pyplot as plt

from gatedmem import GatedMemory

seed = np.random.randint(100)
np.random.seed(seed)
print seed

dim = 32
isi = 1.0
neurons_per_dim = 50
cconv_neurons = 200
sim_time = 60.0

vocab = spa.Vocabulary(dim)
vocab.add('POS', SemanticPointer(dim).unitary())
print np.linalg.norm(vocab['POS'].v)

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
    nengo.Connection(model.clock, model.inv_clock, function=lambda x: abs(x-1))

    model.pos = nengo.Node(output=vocab['POS'].v)
    model.position = spa.State(vocab)
    model.inp = spa.Input()
    model.inp.position=init_pos
    
    model.current = GatedMemory(neurons_per_dim, dim)
    model.next = GatedMemory(neurons_per_dim, dim)
    model.cconv = nengo.networks.CircularConvolution(cconv_neurons, dimensions=dim)

    nengo.Connection(model.inv_clock, model.current.gate)
    nengo.Connection(model.clock, model.next.gate)

    nengo.Connection(model.position.output, model.current.input)
    nengo.Connection(model.current.output, model.cconv.input_a)
    nengo.Connection(model.pos, model.cconv.input_b)
    nengo.Connection(model.cconv.output, model.next.input)
    nengo.Connection(model.next.output, model.current.input)

    cur_probe = nengo.Probe(model.current.output, synapse=0.03)
    next_probe = nengo.Probe(model.next.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(sim_time)
t = sim.trange()

cur_data = sim.data[cur_probe]
next_data = sim.data[next_probe]

def normalized_dot(x, y):
    xn = np.linalg.norm(x)
    if xn != 0:
        x = x / np.linalg.norm(x)
    yn = np.linalg.norm(y)
    if yn != 0:
        y = y / np.linalg.norm(y)
    return np.dot(x, y)

def plot_similarity(similarity, title='Similarity'):
    ymin, ymax = -0.2, 1.5
    plt.figure()
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.plot(t, similarity)
    plt.title(title)
    plt.xlabel('Time')
    plt.xlim(right=t[-1])
    plt.legend(vocab.keys(), loc='upper center')
    plt.show()

def plot_pointers(data, vocab, title='Similarity'):
    plot_similarity(spa.similarity(data, vocab), title)

def plot_vector_length(data, title='Length'):
    data = np.asarray([np.linalg.norm(x) for x in data])
    plot_similarity(data, title)

def plot_index_sim(cur, next):
    data = np.asarray([normalized_dot(cur[i], next[i]) for i in range(0, len(cur))])
    plot_similarity(data)

plot_index_sim(cur_data, next_data)
plot_vector_length(cur_data, 'cl')
plot_vector_length(next_data, 'nl')
