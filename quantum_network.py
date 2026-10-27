import math

import numpy as np
import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
try:
    from qiskit.circuit.library import real_amplitudes
except ImportError:                       # older Qiskit: the class has the same keyword arguments
    from qiskit.circuit.library import RealAmplitudes as real_amplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from neural_network import CorrectionNetwork


def build_qnn(n_qubits, reps, entanglement):
    inputs = ParameterVector('x', n_qubits)
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(inputs[q], q)
    ansatz = real_amplitudes(n_qubits, entanglement=entanglement, reps=reps)
    qc.compose(ansatz, inplace=True)

    observables = [SparsePauliOp.from_sparse_list([('Z', [q], 1.0)], num_qubits=n_qubits)
                   for q in range(n_qubits)]
    return EstimatorQNN(
        circuit=qc,
        input_params=list(inputs),
        weight_params=list(ansatz.parameters),
        observables=observables,
        input_gradients=True,             # required so gradients reach the classical encoder
    )


class QuantumCorrectionNetwork(CorrectionNetwork):
    """
    Same prediction model as CorrectionNetwork (cost_hat = c - sum_i x_i * gamma_iq * v_iq, gamma in (0, 2),
    gamma = 1 at initialisation), but the configuration-dependent term is
    encoder -> QNN -> classical MLP -> head.

    quantum=False swaps the QNN for a classical n_qubits -> n_qubits layer, for a like-for-like ablation.
    """

    def __init__(self, benefits, baseline, n_qubits=4, reps=2, entanglement='reverse_linear',
                 hidden=(32,), quantum=True, seed=0):
        super().__init__(benefits, baseline, hidden=())     # sets V, c, theta; no classical trunk
        self.n_qubits = n_qubits
        self.encoder = nn.Linear(self.n_cand, n_qubits)

        if quantum:
            qnn = build_qnn(n_qubits, reps, entanglement)
            init = 0.1 * np.random.default_rng(seed).standard_normal(qnn.num_weights)
            self.qlayer = TorchConnector(qnn, initial_weights=init)
        else:
            self.qlayer = nn.Sequential(nn.Linear(n_qubits, n_qubits), nn.Tanh())

        layers, d = [], n_qubits
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.post = nn.Sequential(*layers)
        self.head = nn.Linear(d, self.V.numel())
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def gamma(self, x):                                      # (B, n_cand) -> (B, n_cand, n_tmpl)
        z = self.theta.expand(x.shape[0], -1, -1)
        angles = math.pi * torch.tanh(self.encoder(x))       # (B, n_qubits)
        feats = self.qlayer(angles).to(x.dtype)              # (B, n_qubits)
        z = z + self.head(self.post(feats)).view(-1, self.n_cand, self.n_tmpl)
        return 2 * torch.sigmoid(z)