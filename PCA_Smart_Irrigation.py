import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
ROWS, COLS = 5, 5
MAX_ITER = 100
ALPHA = 0.25

M_curr = np.full((ROWS, COLS), 20.0)
M_targ = np.full((ROWS, COLS), 42.4)
Evap   = np.full((ROWS, COLS), 4.0)

def evaluate_fitness(water_mins, m_curr, m_targ, evap):
    predicted_moisture = m_curr + (1.2 * water_mins) - evap
    return np.abs(m_targ - predicted_moisture)

def get_best_neighbor(r, c, state, fitness):
    best_fit = fitness[r, c]
    best_val = state[r, c]
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if fitness[nr, nc] < best_fit:
                    best_fit = fitness[nr, nc]
                    best_val = state[nr, nc]
    return best_val

state = np.random.uniform(5, 15, (ROWS, COLS))
fitness = evaluate_fitness(state, M_curr, M_targ, Evap)

best_fitness_history = []

for t in range(MAX_ITER):
    best_fitness_history.append(np.min(fitness))
    
    new_state = np.copy(state)
    step_noise = 1.5 * np.exp(-0.06 * t)
    
    for r in range(ROWS):
        for c in range(COLS):
            x_best_neighbor = get_best_neighbor(r, c, state, fitness)
            candidate = state[r, c] + ALPHA * (x_best_neighbor - state[r, c]) + np.random.uniform(-step_noise, step_noise)
            candidate = np.clip(candidate, 0, 30)
            
            if evaluate_fitness(candidate, M_curr[r, c], M_targ[r, c], Evap[r, c]) < fitness[r, c]:
                new_state[r, c] = candidate

    state = new_state
    fitness = evaluate_fitness(state, M_curr, M_targ, Evap)

print("Optimal Watering Schedule (Minutes per Garden Zone):")
print(np.round(state, 2))
print(f"\nInitial Best Fitness: {best_fitness_history[0]:.4f}")
print(f"Final Best Fitness:   {best_fitness_history[-1]:.5f}")

plt.figure(figsize=(7, 4.5))
plt.semilogy(
    range(MAX_ITER),
    best_fitness_history,
    color='#1d248b',
    linewidth=2.2,
    marker='o',
    markersize=6,
    markevery=12
)
plt.title("Convergence", fontsize=15, fontweight='bold')
plt.xlabel("Iteration", fontsize=12, fontweight='bold')
plt.ylabel("Best Fitness f(x)", fontsize=12, fontweight='bold')
plt.xlim(0, MAX_ITER)
plt.ylim(1e-3, 2e1)
plt.grid(True, which="major", linestyle="-", alpha=0.3)
plt.tight_layout()
plt.show()
