import numpy as np

np.random.seed(42)

HOURS = 24
NUM_WOLVES = 40
MAX_ITER = 150

LOAD_DEMAND = np.array([40, 38, 35, 34, 36, 42, 55, 70, 80, 85, 82, 78,
                        75, 76, 80, 88, 95, 110, 115, 105, 90, 75, 60, 48], dtype=float)

SOLAR_MAX = np.array([0, 0, 0, 0, 0, 5, 18, 35, 55, 70, 80, 85,
                      82, 75, 60, 42, 22, 6, 0, 0, 0, 0, 0, 0], dtype=float)

WIND_MAX = np.array([25, 28, 30, 32, 30, 26, 22, 18, 15, 14, 16, 18,
                     20, 22, 24, 26, 28, 32, 35, 34, 30, 28, 26, 25], dtype=float)

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

wolves = np.random.uniform(LB, UB, (NUM_WOLVES, DIM))
alpha_pos = np.zeros(DIM)
alpha_score = float("inf")
alpha_metrics = {}

beta_pos = np.zeros(DIM)
beta_score = float("inf")

delta_pos = np.zeros(DIM)
delta_score = float("inf")

for t in range(MAX_ITER):
    for i in range(NUM_WOLVES):
        wolves[i] = np.clip(wolves[i], LB, UB)
        fitness, metrics = evaluate_schedule(wolves[i])

        if fitness < alpha_score:
            delta_score, delta_pos = beta_score, beta_pos.copy()
            beta_score, beta_pos = alpha_score, alpha_pos.copy()
            alpha_score, alpha_pos = fitness, wolves[i].copy()
            alpha_metrics = metrics
        elif fitness < beta_score:
            delta_score, delta_pos = beta_score, beta_pos.copy()
            beta_score, beta_pos = fitness, wolves[i].copy()
        elif fitness < delta_score:
            delta_score, delta_pos = fitness, wolves[i].copy()

    a = 2.0 - t * (2.0 / MAX_ITER)

    for i in range(NUM_WOLVES):
        for j in range(DIM):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            X1 = alpha_pos[j] - A1 * abs(C1 * alpha_pos[j] - wolves[i, j])

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            X2 = beta_pos[j] - A2 * abs(C2 * beta_pos[j] - wolves[i, j])

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            X3 = delta_pos[j] - A3 * abs(C3 * delta_pos[j] - wolves[i, j])

            wolves[i, j] = (X1 + X2 + X3) / 3.0

print("--- Renewable Energy Optimization Results (GWO) ---")
print(f"Minimum Operating Cost : ${alpha_metrics['min_op_cost']:.2f}")
print(f"Grid Electricity       : {alpha_metrics['grid_elec']:.2f} kWh")
print(f"Renewable Utilization  : {alpha_metrics['renewable_util']:.2f} %")
print(f"Grid Dependency        : {alpha_metrics['grid_depend']:.2f} %")
print(f"Curtailed Energy       : {alpha_metrics['curtailed_energy']:.2f} kWh")
print(f"Final Battery Cost     : ${alpha_metrics['final_battery_cost']:.2f}")
