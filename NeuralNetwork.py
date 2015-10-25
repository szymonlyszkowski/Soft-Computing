from ActivationFunction import ActivationFunction
from Neuron import Neuron

__author__ = 'szymonidas'


class NeuralNetwork:
    def __init__(self):
        self.training_set = [3, 2, 4]
        self.initial_weights = [1,1,1]
        self.neural_network_argument = None

    def train_neural_network(self):
        activation_function = ActivationFunction()
        neuron = Neuron()
        neuron.initialize_neuron(self.initial_weights, self.training_set)
        self.neural_network_argument = neuron.compute_argument_value_for_activation_function()

        while self.neural_network_argument != 19.0:
            if neuron.check_if_activate(self.neural_network_argument):
                self.neural_network_argument = activation_function.identity_function(self.neural_network_argument)
                print 'argument value in loop %s ' % self.neural_network_argument
                neuron.apply_new_weights(19, self.neural_network_argument)
            self.neural_network_argument = neuron.compute_argument_value_for_activation_function()

        print "Neural network trained weights are: %s" % neuron.weights


