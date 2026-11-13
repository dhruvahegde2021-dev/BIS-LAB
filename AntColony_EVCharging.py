import numpy as np

np.random.seed(42)

HOURS = 24
NUM_EVS = 15
NUM_ANTS = 30
MAX_ITER = 100

ALPHA = 1.0
BETA = 2.0
RHO = 0.2
Q = 100.0

TARIFF = np.array([
    0.08, 0.08, 0.07, 0.07, 0.08, 0.10,
    0.14, 0.18, 0.20, 0.16, 0.14, 0.13,
    0.12, 0.12, 0.13, 0.15, 0.22, 0.28,
    0.30, 0.28, 0.22, 0.16, 0.12, 0.09
], dtype=float)

RENEWABLE_AVAIL = np.array([
    5.0, 5.0, 4.0, 4.0, 5.0, 8.0,
    15.0, 25.0, 35.0, 45.0, 55.0, 60.0,
    62.0, 58.0, 50.0, 40.0, 25.0, 12.0,
    6.0, 5.0, 5.0, 4.0, 4.0, 5.0
], dtype=float)

BASE_LOAD = np.array([
    20.0, 18.0, 17.0, 17.0, 19.0, 24.0,
    32.0, 40.0, 45.0, 42.0, 40.0, 38.0,
    39.0, 40.0, 42.0, 46.0, 55.0, 65.0,
    68.0, 62.0, 50.0, 38.0, 28.0, 22.0
], dtype=float)

EV_ARRIVAL = np.array([0, 1, 2, 6, 7, 8, 8, 9, 10, 12, 14, 16, 17, 18, 19])
EV_DURATION = np.array([3, 4, 3, 2, 4, 3, 2, 3, 4, 2, 3, 3, 4, 3, 2])
CHARGER_RATE = 7.5
GRID_LIMIT = 85.0

pheromone = np.ones((NUM_EVS, HOURS))

heuristic = np.zeros((NUM_EVS, HOURS))
for e in range(NUM_EVS):
    dur = EV_DURATION[e]
    arr = EV_ARRIVAL[e]
    for h in range(HOURS):
        if h >= arr and (h + dur) <= HOURS:
            avg_price = np.mean(TARIFF[h:h + dur])
            avg_re = np.mean(RENEWABLE_AVAIL[h:h + dur])
            heuristic[e, h] = (1.0 / avg_price) + (0.05 * avg_re)
        else:
            heuristic[e, h] = 1e-6

def evaluate_ev_schedule(start_hours):
    ev_load = np.zeros(HOURS)
    total_wait = 0.0

    for e in range(NUM_EVS):
        s = start_hours[e]
        d = EV_DURATION[e]
        ev_load[s:s + d] += CHARGER_RATE
        total_wait += max(0, s - EV_ARRIVAL[e])

    total_load = BASE_LOAD + ev_load
    re_used = np.minimum(total_load, RENEWABLE_AVAIL)
    grid_elec = np.maximum(0.0, total_load - RENEWABLE_AVAIL)

    ev_grid_share = np.maximum(0.0, ev_load - np.maximum(0.0, RENEWABLE_AVAIL - BASE_LOAD))
    charging_cost = np.sum(ev_grid_share * TARIFF) + np.sum((ev_load - ev_grid_share) * 0.03)

    peak_overload = np.sum(np.maximum(0.0, total_load - GRID_LIMIT))
    wait_penalty = total_wait * 0.15
    fitness = charging_cost + (peak_overload * 2.5) + wait_penalty

    re_util = (np.sum(re_used) / np.sum(RENEWABLE_AVAIL)) * 100.0
    grid_depend = (np.sum(grid_elec) / np.sum(total_load)) * 100.0

    metrics = {
        "min_charging_cost": charging_cost,
        "grid_elec": np.sum(grid_elec),
        "renewable_util": re_util,
        "grid_depend": grid_depend,
        "peak_load": np.max(total_load),
        "avg_wait_time": total_wait / NUM_EVS
    }
    return fitness, metrics

best_schedule = None
best_fitness = float("inf")
best_metrics = {}

for it in range(MAX_ITER):
    ant_schedules = []
    ant_fitness = []

    for a in range(NUM_ANTS):
        schedule = np.zeros(NUM_EVS, dtype=int)
        for e in range(NUM_EVS):
            valid_hours = np.arange(EV_ARRIVAL[e], HOURS - EV_DURATION[e] + 1)
            tau_vals = pheromone[e, valid_hours] ** ALPHA
            eta_vals = heuristic[e, valid_hours] ** BETA
            probs = tau_vals * eta_vals
            probs /= np.sum(probs)
            schedule[e] = np.random.choice(valid_hours, p=probs)

        fit, met = evaluate_ev_schedule(schedule)
        ant_schedules.append(schedule)
        ant_fitness.append(fit)

        if fit < best_fitness:
            best_fitness = fit
            best_schedule = schedule.copy()
            best_metrics = met

    pheromone *= (1.0 - RHO)
    for a in range(NUM_ANTS):
        deposit = Q / ant_fitness[a]
        for e in range(NUM_EVS):
            pheromone[e, ant_schedules[a][e]] += deposit

print("--- EV Charging Optimization Results (Ant Colony Optimization) ---")
print(f"Minimum Charging Cost  : ${best_metrics['min_charging_cost']:.2f}")
print(f"Grid Electricity       : {best_metrics['grid_elec']:.2f} kWh")
print(f"Renewable Utilization  : {best_metrics['renewable_util']:.2f} %")
print(f"Grid Dependency        : {best_metrics['grid_depend']:.2f} %")
print(f"Peak Station Load      : {best_metrics['peak_load']:.2f} kW")
print(f"Average Waiting Time   : {best_metrics['avg_wait_time']:.2f} hrs")
