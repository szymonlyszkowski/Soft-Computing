import random
from ActivationFunction import ActivationFunction
from Neuron import Neuron


class NeuralNetwork:
    __DESIRED_RESULT = 19

    def __init__(self):
        self.training_set = [3, 2, 4]
        self.initial_weights = ([random.uniform(0, 1) for _ in range(0, 3)])  # +1 for bias weight
        # self.initial_weights = [-1, 0, 0]

    def train_neural_network(self):
        activation_function = ActivationFunction()
        neuron = Neuron()
        neuron.initialize_neuron(self.initial_weights, self.training_set)
        neural_network_argument = neuron.compute_argument_value_for_activation_function()

        while neural_network_argument != self.__DESIRED_RESULT:
            neural_network_argument = activation_function.identity_function(neural_network_argument)
            print 'argument value in loop %s ' % neural_network_argument
            neuron.apply_new_weights(self.__DESIRED_RESULT, neural_network_argument)
            neural_network_argument = neuron.compute_argument_value_for_activation_function()

        print "Neural network trained weights are: %s" % neuron.weights

    def __assert_weights(self, weights):
        return all(isinstance(weight, int) for weight in weights)
