# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# %%
def random_clifford_t_block(qc, q1, q2, add_t, p_t=0.2):
    for q in [q1, q2]:
        r = np.random.randint(4)
        if r == 0:
            qc.h(q)
        elif r == 1:
            qc.s(q)
        elif r == 2:
            qc.h(q)
            qc.s(q)

    if np.random.rand() < 0.5:
        qc.cx(q1, q2)
    else:
        qc.cx(q2, q1)

    if add_t:
        if np.random.rand() < p_t:
            qc.t(q1)
        if np.random.rand() < p_t:
            qc.t(q2)


def build_circuit_2d(Lx, Ly, d, add_t, p_t=0.2):
    N = Lx * Ly
    qc = QuantumCircuit(N, N)

    def idx(x, y):
        return y * Lx + x

    for layer in range(d):
        direction = layer % 2
        offset = (layer // 2) % 2

        gates_added = False
        if direction == 0:
            for y in range(Ly):
                for x in range(offset, Lx - 1, 2):
                    random_clifford_t_block(qc, idx(x, y), idx(x + 1, y), add_t=add_t, p_t=p_t)
                    gates_added = True
        else:
            for x in range(Lx):
                for y in range(offset, Ly - 1, 2):
                    random_clifford_t_block(qc, idx(x, y), idx(x, y + 1), add_t=add_t, p_t=p_t)
                    gates_added = True

        if gates_added:
            qc.barrier()

    qc.measure(range(N), range(N))
    return qc


def projected_collision_prob(counts, n_A):
    total_shots = sum(counts.values())

    ab_counts = defaultdict(lambda: defaultdict(int))
    a_counts = defaultdict(int)

    for bitstring, count in counts.items():
        bitstring = bitstring.replace(' ', '')
        a_bits = bitstring[-n_A:]
        b_bits = bitstring[:-n_A]
        ab_counts[a_bits][b_bits] += count
        a_counts[a_bits] += count

    cp = 0.0
    for a_bits, b_dict in ab_counts.items():
        p_a = a_counts[a_bits] / total_shots
        n_a = a_counts[a_bits]
        cond_cp = sum((c / n_a) ** 2 for c in b_dict.values())
        cp += p_a * cond_cp

    return cp

# %%
# Fixed N=6x6=36, fixed depth=10, vary shots
L = 6
d = 10
add_t = True
samples_per_shots = 10
shot_values = [100, 250, 500, 1000, 2000, 5000, 10000]
output_dir = "results/vary_shots"
os.makedirs(output_dir, exist_ok=True)

N = L * L
n_A = N // 2
n_B = N - n_A

sim = AerSimulator(
    method="matrix_product_state",
    mps_sample_measure_algorithm="mps_apply_measure",
    matrix_product_state_max_bond_dimension=64,
    matrix_product_state_truncation_threshold=1e-8,
    max_parallel_threads=0,
    mps_omp_threads=0
)

avg_prob, std_prob, completed_shots = [], [], []

for shots in shot_values:
    print(f"shots={shots}, N={N}, d={d}, n_A={n_A}, n_B={n_B}")
    vals = []

    for _ in range(samples_per_shots):
        qc = build_circuit_2d(L, L, d, add_t, p_t=0.15)
        tqc = transpile(qc, basis_gates=['h', 's', 'sdg', 't', 'tdg', 'cx', 'measure'], optimization_level=0)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        vals.append((2 ** n_B) * projected_collision_prob(counts, n_A))

    avg_prob.append(np.mean(vals))
    std_prob.append(np.std(vals))
    completed_shots.append(shots)

    fig, ax = plt.subplots()
    ax.errorbar(completed_shots, avg_prob, yerr=std_prob, marker='o', capsize=4, label=f"N={N}, d={d}")
    ax.set_xlabel("shots")
    ax.set_ylabel(r"$2^{n_B} \sum_x p(x) \sum_y p(y|x)^2$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"2D Brickwork: Projected Ensemble CP vs Shots  (N={N}, d={d})")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"vary_shots_N{N}_d{d}_upto_{shots}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved vary_shots_N{N}_d{d}_upto_{shots}.png")

plt.errorbar(completed_shots, avg_prob, yerr=std_prob, marker='o', capsize=4, label=f"N={N}, d={d}")
plt.xlabel("shots")
plt.ylabel(r"$2^{n_B} \sum_x p(x) \sum_y p(y|x)^2$")
plt.xscale('log')
plt.yscale('log')
plt.title(f"2D Brickwork: Projected Ensemble CP vs Shots  (N={N}, d={d})")
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.show()


