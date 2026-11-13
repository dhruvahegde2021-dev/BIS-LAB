import numpy as np
import math

np.random.seed(42)

HOURS = 24
NUM_NESTS = 40
MAX_ITER = 150
PA = 0.25
BETA = 1.5

LOAD_DEMAND = np.array([
    35, 32, 30, 28, 30, 38, 48, 60, 72, 78, 80, 82,
    85, 83, 80, 76, 75, 82, 90, 95, 88, 75, 58, 42
], dtype=float)

SOLAR_MAX = np.array([
    0, 0, 0, 0, 0, 2, 10, 25, 42, 58, 70, 78,
    80, 75, 65, 50, 32, 12, 2, 0, 0, 0, 0, 0
], dtype=float)

WIND_MAX = np.array([
    22, 24, 25, 26, 25, 22, 18, 15, 12, 10, 12, 14,
    15, 16, 18, 20, 22, 25, 28, 30, 28, 26, 24, 23
], dtype=float)

BATT_CAP = 100.0
BATT_MAX_RATE = 25.0
SOLAR_COST = 0.04
WIND_COST = 0.03
GRID_COST = 0.18
BATT_DEG_COST = 0.08
CURTAIL_PENALTY = 0.02

DIM = HOURS * 3
LB = np.zeros(DIM)
UB = np.concatenate([SOLAR_MAX, WIND_MAX, np.full(HOURS, BATT_MAX_RATE)])
LB[2 * HOURS:] = -BATT_MAX_RATE

def evaluate_schedule(pos):
    solar_used = pos[:HOURS]
    wind_used = pos[HOURS:2 * HOURS]
    batt_power = pos[2 * HOURS:]

    soc = 50.0
    actual_batt = np.zeros(HOURS)
    soc_penalty = 0.0

    for h in range(HOURS):
        p = batt_power[h]
        if p > 0:
            p = min(p, soc - 10.0)
            p = max(0.0, p)
        else:
            p = max(p, -(BATT_CAP - soc))
            p = min(0.0, p)
        soc -= p
        actual_batt[h] = p
        if soc < 10.0 or soc > BATT_CAP:
            soc_penalty += 500.0

    renewable_supplied = solar_used + wind_used
    net_demand = LOAD_DEMAND - (renewable_supplied + actual_batt)
    grid_elec = np.maximum(0.0, net_demand)
    excess_power = np.maximum(0.0, -net_demand)

    available_renewables = SOLAR_MAX + WIND_MAX
    curtailed_energy = np.sum(available_renewables - renewable_supplied) + np.sum(excess_power)

    op_cost_renewables = np.sum(solar_used * SOLAR_COST + wind_used * WIND_COST)
    op_cost_grid = np.sum(grid_elec * GRID_COST)
    final_battery_cost = np.sum(np.abs(actual_batt) * BATT_DEG_COST)
    total_op_cost = op_cost_renewables + op_cost_grid + final_battery_cost

    fitness = total_op_cost + (curtailed_energy * CURTAIL_PENALTY) + soc_penalty

    total_available_re = np.sum(available_renewables)
    total_used_re = np.sum(renewable_supplied) - np.sum(excess_power)
    renewable_util = (total_used_re / total_available_re) * 100.0
    grid_depend = (np.sum(grid_elec) / np.sum(LOAD_DEMAND)) * 100.0

    metrics = {
        "min_op_cost": total_op_cost,
        "grid_elec": np.sum(grid_elec),
        "renewable_util": renewable_util,
        "grid_depend": grid_depend,
        "curtailed_energy": curtailed_energy,
        "final_battery_cost": final_battery_cost
    }
    return fitness, metrics

def levy_flight(dim):
    num = math.gamma(1 + BETA) * math.sin(math.pi * BETA / 2)
    den = math.gamma((1 + BETA) / 2) * BETA * (2 ** ((BETA - 1) / 2))
    sigma_u = (num / den) ** (1 / BETA)
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / BETA))

nests = np.random.uniform(LB, UB, (NUM_NESTS, DIM))
fitness_vals = np.zeros(NUM_NESTS)
metrics_list = [None] * NUM_NESTS

for i in range(NUM_NESTS):
    fitness_vals[i], metrics_list[i] = evaluate_schedule(nests[i])

best_idx = np.argmin(fitness_vals)
best_nest = nests[best_idx].copy()
best_fitness = fitness_vals[best_idx]
best_metrics = metrics_list[best_idx]

for t in range(MAX_ITER):
    for i in range(NUM_NESTS):
        step = 0.05 * levy_flight(DIM) * (nests[i] - best_nest)
        new_nest = np.clip(nests[i] + step, LB, UB)
        new_fit, new_met = evaluate_schedule(new_nest)

        j = np.random.randint(0, NUM_NESTS)
        if new_fit < fitness_vals[j]:
            nests[j] = new_nest
            fitness_vals[j] = new_fit
            metrics_list[j] = new_met

    abandon_mask = np.random.rand(NUM_NESTS, DIM) < PA
    perm1 = np.random.permutation(NUM_NESTS)
    perm2 = np.random.permutation(NUM_NESTS)
    step_size = np.random.rand() * (nests[perm1] - nests[perm2])

    for i in range(NUM_NESTS):
        if np.any(abandon_mask[i]):
            cand = np.where(abandon_mask[i], nests[i] + step_size[i], nests[i])
            cand = np.clip(cand, LB, UB)
            cand_fit, cand_met = evaluate_schedule(cand)
            if cand_fit < fitness_vals[i]:
                nests[i] = cand
                fitness_vals[i] = cand_fit
                metrics_list[i] = cand_met

    curr_best_idx = np.argmin(fitness_vals)
    if fitness_vals[curr_best_idx] < best_fitness:
        best_fitness = fitness_vals[curr_best_idx]
        best_nest = nests[curr_best_idx].copy()
        best_metrics = metrics_list[curr_best_idx]

print("--- Renewable Energy Optimization Results (Cuckoo Search) ---")
print(f"Minimum Operating Cost : ${best_metrics['min_op_cost']:.2f}")
print(f"Grid Electricity       : {best_metrics['grid_elec']:.2f} kWh")
print(f"Renewable Utilization  : {best_metrics['renewable_util']:.2f} %")
print(f"Grid Dependency        : {best_metrics['grid_depend']:.2f} %")
print(f"Curtailed Energy       : {best_metrics['curtailed_energy']:.2f} kWh")
print(f"Final Battery Cost     : ${best_metrics['final_battery_cost']:.2f}")
