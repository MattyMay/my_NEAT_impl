from population import Population
from genome import Genome

class Config:
    def __init__(self):
        self.num_inputs = 2
        self.num_outputs = 1
        self.pop_size = 25

        self.max_weight = 10
        self.weight_mutation_rate = 0.4
        self.max_weight_mutation = 3
        self.weight_replace_rate = 0.2
        self.add_node_rate = 0.2
        self.add_connection_rate = 0.2

        self.topological_comp_coef = 1
        self.weight_comp_coef = 0.5
        self.comp_threshold = 2

        self.reproduce_rate = 0.5

        self.reproduce_rate = 0.5


def fitness_func(genome):
    nn = genome.build_neural_net()
    inputs = [[0,0],[0,1],[1,1],[1,0]]
    correct_outputs = [0, 1, 0, 1]
    outputs = []
    for i in range(4):
        outputs.append(genome.activate(nn, inputs[i]))
    
    diff_sum = 0
    for i in range(4):
        diff_sum += abs(outputs[i][0] - correct_outputs[i])

    return 1/diff_sum

population = Population(Config(), fitness_func)
for i in range(100):
    population.do_timestep()
    population.print_stats()
    print("Generation: {} Best fitness {}".format(i, population.best.fitness))

