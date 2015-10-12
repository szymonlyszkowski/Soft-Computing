from ActivationFunction import ActivationFunction
from Neuron import Neuron

__author__ = 'szymonidas'


class NeuralNetwork:
    def __init__(self):
        self.training_set = [3, 2, 1]
        self.initial_weights = [0, 2, 2]

    def train_neural_network(self):
        activation_function = ActivationFunction()
        neuron = Neuron()
        neuron.initialize_neuron(self.initial_weights, self.training_set)
        argument = neuron.compute_argument_value_for_activation_function()
        while 19 != argument:
            #if neuron.check_if_activate(argument):
            argument = activation_function.identity_function(argument)
            neuron.apply_new_weights(19, argument)
