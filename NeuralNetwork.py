import random
from ActivationFunction import ActivationFunction
from Neuron import Neuron


class NeuralNetwork:
    __DESIRED_RESULT = 19

    def __init__(self):
        self.results = [19, 13, 13]
        self.training_set = [3, 2, 4]
        self.training_set2 = [2, 1, 3]
        self.training_set3 = [1, 3, 2]
        self.training_set_result = [self.training_set, self.training_set2, self.training_set3]
        self.initial_weights = ([random.uniform(0, 1) for _ in range(0, 3)])  # +1 for bias weight
        # self.initial_weights = [0.1, 0, 0]

    def train_neural_network(self):
        activation_function = ActivationFunction()
        neuron = Neuron()
        neuron.initialize_neuron(self.initial_weights, self.training_set)
        #

        for iteration in range(0, 100000):
            for training_set_index,training_set in enumerate(self.training_set_result):
                neuron.apply_training_set(training_set)
                neural_network_argument = neuron.compute_argument_value_for_activation_function()
                activation_function.identity_function(neural_network_argument)
                neuron.apply_new_weights(self.results[training_set_index], neural_network_argument)
                #neural_network_argument = neuron.compute_argument_value_for_activation_function()
                print 'argument value in loop %s ' % neural_network_argument

        print "Neural network trained weights are: %s" % neuron.weights

    def __assert_weights(self, weights):
        return all(isinstance(weight, int) for weight in weights)
