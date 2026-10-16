import random

TASKS = [
    {"id": 1, "duration": 4},
    {"id": 2, "duration": 2},
    {"id": 3, "duration": 8},
    {"id": 4, "duration": 5},
    {"id": 5, "duration": 3},
    {"id": 6, "duration": 6},
]

# Adding multi-processor execution constraint
NUM_PROCESSORS = 2

POPULATION_SIZE = 50
MUTATION_RATE = 0.15
GENERATIONS = 100


def create_individual():
    ind = [task["id"] for task in TASKS]
    random.shuffle(ind)
    return ind


def simulate_schedule(individual):
    """
    Simulates a greedy allocation of the task sequence across parallel processors.
    """
    processor_free_times = [0] * NUM_PROCESSORS
    task_map = {t["id"]: t["duration"] for t in TASKS}
    
    # Track which task goes to which processor for clean reporting
    processor_assignments = {i: [] for i in range(NUM_PROCESSORS)}

    for task_id in individual:
        # Find the processor that becomes free earliest
        best_proc = min(range(NUM_PROCESSORS), key=lambda p: processor_free_times[p])
        
        # Assign task to that processor
        processor_assignments[best_proc].append((task_id, task_map[task_id]))
        processor_free_times[best_proc] += task_map[task_id]

    # The makespan is the maximum time taken among all processors
    makespan = max(processor_free_times)
    return makespan, processor_assignments


def calculate_fitness(individual):  
    makespan, _ = simulate_schedule(individual)
    # Minimize makespan by maximizing the inverse
    return 1.0 / makespan


def selection(population):
    tournament_size = 3
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda ind: calculate_fitness(ind), reverse=True)
    return tournament[0]


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start : end + 1] = parent1[start : end + 1]

    ptr = 0
    for item in parent2:
        if item not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = item

    return child


def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


def genetic_algorithm_scheduling():
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        population.sort(key=lambda ind: calculate_fitness(ind), reverse=True)

        new_population = []
        new_population.extend(population[:2])

        while len(new_population) < POPULATION_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    population.sort(key=lambda ind: calculate_fitness(ind), reverse=True)
    best_schedule = population[0]
    best_makespan, assignments = simulate_schedule(best_schedule)

    return best_schedule, best_makespan, assignments


if __name__ == "__main__":
    optimal_schedule, makespan, assignments = genetic_algorithm_scheduling()
    
    print("--- Optimization Results ---")
    print("Optimal Task Sequence Queue:", optimal_schedule)
    print(f"Minimum Makespan (Total Time): {makespan} units\n")
    
    print("--- Processor Allocation Breakdown ---")
    for proc_id, tasks in assignments.items():
        task_str = ", ".join([f"Task {tid} ({dur}u)" for tid, dur in tasks])
        total_proc_time = sum(dur for _, dur in tasks)
        print(f"Processor {proc_id}: [{task_str}] -> Total Active Time: {total_proc_time} units")
