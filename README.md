# my_NEAT_impl
This is a work in progress. The goal is to create a simple approximation of the
algorithm described in *Evolving Neural Networks throughAugmenting Topologies* by
Stanley and Miikkulainen. Some of the interface design is loosely inspired by my
previous use of the *NEAT-Python* library (https://neat-python.readthedocs.io),
although the implementation is not based off of that source code.

Currently I believe that encoding/decoding of genomes works as expected.
Mutation operators should also be working, and I think that crossover works as
well. However the code that handles the population as a whole (and specifically
speciation) is not working yet.
