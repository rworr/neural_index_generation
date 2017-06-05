import nengo
import numpy as np
from nengo import spa
import matplotlib.pyplot as plt

from helpers import output_similarities_to_file as dump

dim = 512
isi = 0.5

vocab = spa.Vocabulary(dim)
vocab.parse('POS')
vocab.add('NEXT', vocab.create_pointer(unitary=True))

for i in range(2, 61):
    vocab.add('POS%d' % i, vocab.parse('POS%d * NEXT' % (i-1)))

def position(t):
    return 'POS%d' % ((t // isi) + 1)

with spa.SPA('Ideal Indexing', vocabs=[vocab], seed=1) as model:
    model.current = spa.State(dimensions=dim)
    model.input = spa.Input(current=position)

    model.prev1 = spa.State(dim)
    model.prev2 = spa.State(dim)
    model.prev3 = spa.State(dim)
    model.prev4 = spa.State(dim)
    model.prev5 = spa.State(dim)
    
    cortical_actions = spa.Actions(
        'prev1 = current * ~NEXT',
        'prev2 = prev1 * ~NEXT',
        'prev3 = prev2 * ~NEXT',
        'prev4 = prev3 * ~NEXT',
        'prev5 = prev4 * ~NEXT',
    )
    model.cortical = spa.Cortical(cortical_actions)

    current_probe = nengo.Probe(model.current.output, synapse=0.03, label="current")
    prev1_probe = nengo.Probe(model.prev1.output, synapse=0.03, label="prev1")
    prev2_probe = nengo.Probe(model.prev2.output, synapse=0.03, label="prev2")
    prev3_probe = nengo.Probe(model.prev3.output, synapse=0.03, label="prev3")
    prev4_probe = nengo.Probe(model.prev4.output, synapse=0.03, label="prev4")
    prev5_probe = nengo.Probe(model.prev5.output, synapse=0.03, label="prev5")
    prev_probes = [prev1_probe, prev2_probe, prev3_probe, prev4_probe, prev5_probe]

with nengo.Simulator(model) as sim:
    sim.run(30.1)
t = sim.trange()

dump(sim, vocab)
with open('prev_sim.csv', 'w') as outfile:
    outfile.write("t,%s\n" % ','.join([p.label for p in prev_probes]))
    cur_data = sim.data[current_probe]
    for i in range(0, len(t)):
        prev_sim = [np.dot(cur_data[i], sim.data[p][i]) for p in prev_probes]
        outfile.write("%f,%s\n" % (t[i], ','.join([str(s) for s in prev_sim])))
