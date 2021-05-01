from genome import Genome
from bisect import bisect_left
from random import random


class Species:
    def __init__(self, genome, id, stagnation=0, fitness=None):
        self.stagnation = stagnation
        self.genomes = [genome]
        self.fitness = genome.fitness if not fitness else fitness
        self.id = id

    def calculate_fitness(self):
        sum = 0
        for genome in self.genomes:
            sum += genome.fitness
        new_fitness = sum / len(self.genomes)
        if new_fitness <= self.fitness:
            self.stagnation += 1
        self.fitness = new_fitness

    def reproduce(self, new_size, reproduce_rate):
        self.genomes.sort(reverse=True, key=lambda gen: gen.fitness)
        new_genomes = []
        count = 0
        num_candidates = len(self.genomes) // (1/reproduce_rate)
        while count < new_size:
            gen1 = random.choice(self.genomes[:num_candidates])
            gen2 = random.choice(self.genomes[:num_candidates])
            new_genomes.append(gen1.recombine(gen2))
            count += 1
        return new_genomes


class Population:
    def __init__(self, config, fitness_func):
        self.config = config
        self.fitness_func = fitness_func
        self.species = []
        self.innovation = config.num_inputs * config.num_outputs + 1
        self.best = None

        # initialize poulation
        for _ in config.pop_size:
            genome = Genome(config)
            genome.fitness = self.fitness_func(genome)

            if not self.species:
                self.species.append(Species(genome, 1))
            else:
                for spec in self.species:
                    diffs = genome.get_distance(spec[0])
                    if (diffs[0] * config.topological_comp_coef
                            + diffs[1] * config.weight_comp_coef
                            < config.comp_threshold):
                        spec.append[genome]
                        break
                    else:
                        new_spec_id = self.species[-1].id + 1
                        self.species.append(Species(genome, new_spec_id))

    def do_timestep(self):
        self.reproduce()
        self.mutate()
        self.evaluate_fitnesses()

    def mutate(self):
        for spec in self.species:
            for genome in spec:
                self.innovation = genome.mutate(self.innovation)

    def reproduce(self):
        # get new orgs
        new_pop = []
        sum_adj_fit = sum(spec.fitness for spec in self.species)
        for spec in self.species:
            num_children = spec.fitness // sum_adj_fit
            new_pop.extend(spec.reproduce(num_children, self.config.reproduce))

        # put into species
        self.classify_species(new_pop)

    def classify_species(self, new_pop):
        new_species = []
        for genome in new_pop:
            is_new = True
            for spec in self.species:
                diffs = genome.get_distance(spec[0])
                if (diffs[0] * self.config.topological_comp_coef
                        + diffs[1] * self.config.weight_comp_coef
                        < self.config.comp_threshold):
                    # check if species already exists in new list
                    # TODO: this should be a binary search but i'm
                    #       lazy right now
                    is_new = False
                    found = False
                    for s in new_species:
                        if s.id == spec.id:
                            found = True
                            s.append(genome)
                            break
                    if not found:
                        new_species.append(
                            Species(genome,
                                    spec.id,
                                    stagnation=spec.stagnation,
                                    fitness=spec.fitness))
                    break
            # if is_new, create new species with new id
            if is_new:
                new_species.append(Species(genome, self.species[-1].id + 1))

        # calculate new adjusted fitnesses for each species
        for spec in new_species:
            spec.calculate_fitness()
        self.species = new_species

    def evaluate_fitnesses(self):
        for spec in self.species:
            for genome in spec.genomes:
                genome.fitness = self.fitness_func(genome)
                if self.best == None or genome.fitness > self.best.fitness:
                    self.best = genome
