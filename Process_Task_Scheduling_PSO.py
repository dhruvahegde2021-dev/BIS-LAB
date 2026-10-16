import numpy as np
import random

# --- EXACT SAME INPUT DATA ---
TASKS = [
    {"id": 1, "duration": 4},
    {"id": 2, "duration": 2},
    {"id": 3, "duration": 8},
    {"id": 4, "duration": 5},
    {"id": 5, "duration": 3},
    {"id": 6, "duration": 6},
]

NUM_PROCESSORS = 2
num_tasks = len(TASKS)
task_durations = [t["duration"] for t in TASKS]
task_ids = [t["id"] for t in TASKS]

# --- PSO HYPERPARAMETERS ---
SWARM_SIZE = 50
MAX_ITERATIONS = 100
W = 0.7   # Inertia weight
C1 = 1.5  # Cognitive parameter (personal memory pull)
C2 = 1.5  # Social parameter (global swarm pull)

def calculate_makespan(position):
    """
    Evaluates fitness by mapping continuous positions to discrete task orderings.
    Uses argsort to convert continuous positions into a unique permutation sequence.
    """
    # Convert continuous position values into an ordered task sequence index
    sequence_indices = np.argsort(position)
    
    # Map back to original task configurations based on sorted order
    processor_free_times = [0] * NUM_PROCESSORS
    processor_assignments = {i: [] for i in range(NUM_PROCESSORS)}

    for idx in sequence_indices:
        t_id = task_ids[idx]
        duration = task_durations[idx]
        
        # Greedy allocation to the earliest available processor
        best_proc = min(range(NUM_PROCESSORS), key=lambda p: processor_free_times[p])
        processor_assignments[best_proc].append((t_id, duration))
        processor_free_times[best_proc] += duration

    makespan = max(processor_free_times)
    return makespan, sequence_indices, processor_assignments

# --- INITIALIZE PSO SWARM ---
# Positions represent continuous values used to determine priority orderings
positions = np.random.uniform(-5, 5, size=(SWARM_SIZE, num_tasks))
velocities = np.random.uniform(-1, 1, size=(SWARM_SIZE, num_tasks))

pbest_positions = np.copy(positions)
pbest_makespans = np.zeros(SWARM_SIZE)

for i in range(SWARM_SIZE):
    pbest_makespans[i], _, _ = calculate_makespan(positions[i])

gbest_idx = np.argmin(pbest_makespans)
gbest_position = np.copy(pbest_positions[gbest_idx])
gbest_makespan = pbest_makespans[gbest_idx]

# --- OPTIMIZATION LOOP ---
for iteration in range(MAX_ITERATIONS):
    for i in range(SWARM_SIZE):
        r1 = np.random.rand(num_tasks)
        r2 = np.random.rand(num_tasks)
        
        # Update Velocity vector 
        velocities[i] = (W * velocities[i] + 
                         C1 * r1 * (pbest_positions[i] - positions[i]) + 
                         C2 * r2 * (gbest_position - positions[i]))
        
        # Update Position vector
        positions[i] += velocities[i]
        
        # Evaluate current layout makespan performance
        current_makespan, _, _ = calculate_makespan(positions[i])
        
        # Update Personal Best record tracking
        if current_makespan < pbest_makespans[i]:
            pbest_makespans[i] = current_makespan
            pbest_positions[i] = np.copy(positions[i])
            
            # Update Global Swarm Best record tracking
            if current_makespan < gbest_makespan:
                gbest_makespan = current_makespan
                gbest_position = np.copy(positions[i])

# --- GENERATE FINAL OUTPUT ---
final_makespan, final_sequence_indices, assignments = calculate_makespan(gbest_position)
optimal_sequence = [task_ids[idx] for idx in final_sequence_indices]

print("--- PSO Optimization Results ---")
print("Optimal Task Sequence Queue:", optimal_sequence)
print(f"Minimum Makespan (Total Time): {final_makespan} units\n")

print("--- Processor Allocation Breakdown ---")
for proc_id, tasks in assignments.items():
    task_str = ", ".join([f"Task {tid} ({dur}u)" for tid, dur in tasks])
    total_proc_time = sum(dur for _, dur in tasks)
    print(f"Processor {proc_id}: [{task_str}] -> Total Active Time: {total_proc_time} units")
