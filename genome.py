import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class ConnectionGene:
    def __init__(self, from_id, to_id, weight, innovation, enabled=True):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight
        self.innovation = innovation
        self.enabled = enabled

    def deepcopy(self):
        return ConnectionGene(
            self.from_id,
            self.to_id,
            self.weight,
            self.innovation,
            self.enabled)


class NeuronGene:
    def __init__(self, id, bias):
        self.id = id
        self.bias = bias

    def deepcopy(self):
        return NeuronGene(self.id, self.bias)


class Genome:
    def __init__(self, config, fitness_func, connections=[], neurons=[]):
        self.config = config
        self.fitness_func = fitness_func
        self.connections = connections
        self.neurons = neurons
        self.fitness = None

    ############################################################################
    #######################       PUBLIC        ################################
    ############################################################################

    def mutate(self, curr_innovation):
        """ Calls mutation functions that take care of each mutation operator.
        returns new innovation number."""
        self.__mutate_weights()
        return self.__mutate_topology(curr_innovation)

    def recombine(self, other):
        """ Produces an offspring with self and other as parents.
        This function randomly selects between genes that have the same
        innovation number, and takes the other genes from more fit parent.
        Returns a new genome."""
        new_connection_genes = []

        # get shared genes. Guaranteed to be first genes because of our
        # implementation.
        i = 0
        while (self.connections[i].innovation
               == other.connections[i].innovation):
            if random.random() > 0.5:
                new_connection_genes.append(
                    self.connections[i].deepcopy())
            else:
                new_connection_genes.append(
                    other.connections[i].deepcopy())
            i += 1

        # copy connection genes from more fit genome
        fitter = self if self.fitness > other.fitness else other
        while i < fitter.connections.size():
            new_connection_genes.append(fitter.connections[i])
            i += 1

        # copy neuron genes from more fit genome
        new_neuron_genes = [neuron.deepcopy() for neuron in fitter.neurons]

        return Genome(
            self.config,
            self.fitness_func,
            new_connection_genes,
            new_neuron_genes)

    def activate(self, input):
        """Activates the neural net.
        Requires size of input to match number of input sensors of the genome.
        Returns activation values of output neurons"""
        # build graph from conns and neurons as adjacency list / memo for DP
        graph = self.__build_graph()

        # recursive function to evaluate a single neuron
        def activate_neuron(neuron):
            # [0:num_inputs] reserved for inputs. Just get from argument to this
            # function.
            if neuron < self.config.num_inputs:
                return input[neuron]

            # don't recompute if already computed
            if graph[neuron][0] is not None:
                return graph[neuron][0]

            # get sum of input values (+ bias)
            inp_sum = graph[neuron][1]
            for conn in graph[neuron][2]:
                inp_sum += (self.connections[conn].weight
                            * activate_neuron(self.connections[conn].from_id))

            graph[neuron][0] = sigmoid(inp_sum)
            return graph[neuron][0]

        # evaluate output neurons
        outputs = []
        for i in range(self.config.num_outputs):
            outputs.append(activate_neuron(i + self.config.num_inputs))

        return outputs

    ############################################################################
    #######################       HELPERS       ################################
    ############################################################################

    def __add_connection(self, from_id, to_id, innovation, weight=None):
        """Helper to add a connection with random weight between neurons.
        Requires that it is valid to connect nodes."""

        weight = random.uniform(-1, 1) * \
            self.config.max_weight if weight == None else weight

        new_conn = ConnectionGene(
            from_id=from_id,
            to_id=to_id,
            weight=weight,
            innovation=innovation)
        self.connections.append(new_conn)

    def __mutate_weights(self):
        """Helper to mutate connection weights.
        Takes care of mutating each weight and replacing a weight."""
        # each weight has a chance to mutate
        for conn in self.connections:
            if random.random() < self.config.weight_mutation_rate:
                adj = random.uniform(-1, 1) * self.config.max_weight_mutation

                new_weight = conn.weight + adj

                if new_weight > self.config.max_weight:
                    conn.weight = self.config.max_weight
                if new_weight < -self.config.max_weight:
                    conn.weight = -self.config.max_weight
                else:
                    conn.weight = new_weight

        # chance to entirely replace one weight
        if random.random() < self.config.weight_replace_rate:
            i = random.randint(0, len(self.connections) - 1)
            new_weight = random.uniform(-1, 1) * self.config.max_weight
            self.connections[i].weight = new_weight

    def __mutate_topology(self, curr_innovation):
        """Helper to mutate topology.
        Takes care of adding hidden neurons, and adding new connections.
        Returns the new innovation number."""
        # add node
        if random.random() < self.config.add_node_rate:
            # choose connection and disable it
            curr_innovation += 1
            i = random.randint(0, len(self.connections) - 1)
            self.connections[i].enabled = False

            # add node, make new connections
            new_id = self.neurons[-1].id
            self.neurons.append(NeuronGene(new_id + 1, 1))
            self.__add_connection(
                self.connections[i].from_id,
                new_id,
                curr_innovation,
                weight=1)

            curr_innovation += 1

            self.__add_connection(
                new_id,
                self.connections[i].to_id,
                curr_innovation,
                weight=self.connections[i].weight)

            curr_innovation += 1

        # add connection
        if random.random() < self.config.add_connection_rate:
            ids = [neuron.id for neuron in self.neurons]
            from_id = random.choice(ids)
            ids.remove(from_id)
            to_id = random.choice(ids)
            ids.remove(to_id)

            while ids:
                if not self.__path_exists(to_id, from_id):
                    # won't create loop, add connection
                    self.__add_connection(from_id, to_id, curr_innovation)

                else:
                    # pick new to_id candidate
                    to_id = random.choice(ids)
                    ids.remove(to_id)

        return curr_innovation

    def __path_exists(self, from_id, to_id):
        """Helper to check if a path exists between two neurons.
        Returns true if one does, false otherwise."""
        # this is probably a bad way to do this, but given our genome
        # representation i'm not sure how else to prevent backward connections.
        # We'll just use BFS to see if a path exists from from_id to to_id

        if from_id == to_id:
            return True

        graph = self.__build_graph(for_activate=False)
        queue = []
        queue.append(from_id)
        graph[from_id][0] = True

        while queue:
            n = queue.pop(0)
            if n == to_id:
                return True
            for i in graph[n][1]:
                if graph[i][0] == False:
                    queue.append(i)
                    graph[i][0] = True

        return False

    def __build_graph(self, for_activate=True):
        """Helper that builds a graph as an adjency list out of genome.
        Used for activation and to prevent backwards connections.
        params: if for_activate is set to True, returns graph as dictionary with
                neuron IDs as keys, and 3-tuple values of the form:
                    (activation_value=None, bias, [<connections>])
                that serves as both an adjency list and a memo for efficiently
                calculating activation of each neuron in a recursive manner.

                if for_activate is set to False, returns a graph much the same
                but with values as tuples of the form:
                    (visited=False, [<connections>])
                that can be used for general graph traversal"""
        graph = {}
        adj_list_ind = 2
        for neuron in self.neurons:
            if for_activate:
                graph[neuron.id] = [None, neuron.bias, []]
            else:
                graph[neuron.id] = [False, []]
                adj_list_ind = 1
        for i in range(len(self.connections)):
            if self.connections[i].enabled:
                graph[self.connections[i].to_id][adj_list_ind].append(i)
        return graph


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.num_inputs = 2
            self.num_outputs = 2

    config = Config()
    out1 = NeuronGene(2, -3)
    out2 = NeuronGene(3, -3)
    conn1 = ConnectionGene(0, 2, 0, 0)
    conn2 = ConnectionGene(0, 3, 6, 1)
    conn3 = ConnectionGene(1, 2, 6, 2)
    conn4 = ConnectionGene(1, 3, 0, 3)

    conns = [conn1, conn2, conn3, conn4]
    neurs = [out1, out2]

    def fitness(fit):
        return 0

    genome = Genome(config, fitness, conns, neurs)
    print(genome.activate([1, 1]))
