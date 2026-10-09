import random

TASKS = [
    {"id": 1, "duration": 4},
    {"id": 2, "duration2": 2, "duration": 2},
    {"id": 3, "duration": 8},
    {"id": 4, "duration": 5},
    {"id": 5, "duration": 3},
    {"id": 6, "duration": 6},
]

POPULATION_SIZE = 50
MUTATION_RATE = 0.15
GENERATIONS = 100


def create_individual():
    ind = [task["id"] for task in TASKS]
    random.shuffle(ind)
    return ind


def calculate_fitness(individual):  
    total_time = 0
    task_map = {t["id"]: t["duration"] for t in TASKS}

    for task_id in individual:
        total_time += task_map[task_id]

    return 1.0 / total_time


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
    best_makespan = sum(
        next(t["duration"] for t in TASKS if t["id"] == tid)
        for tid in best_schedule
    )

    return best_schedule, best_makespan


if __name__ == "__main__":
    optimal_schedule, makespan = genetic_algorithm_scheduling()
    print("Optimal Task Sequence (IDs):", optimal_schedule)
    print("Minimum Makespan (Total Time):", makespan)
