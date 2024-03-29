import random
import math
import numpy
from neurons.neuron import Neuron


class AbstractNeuralNetwork(object):
    _KOHONEN_WEIGHT_MIN = 0
    _KOHONEN_WEIGHT_MAX = 255

    _PERCEPTRON_WEIGHT_MIN = -0.5
    _PERCEPTRON_WEIGHT_MAX = 0.5

    def __init__(self, training_set, neurons_amount, weights_amount_in_neuron):
        self.training_set = training_set
        self.network_neurons = self._init_network_with_neurons(neurons_amount)
        self.__initialize_network_neurons_with_random_weights(weights_amount_in_neuron)

    # FOR KOHONEN (0,255)
    def __initialize_network_neurons_with_random_weights(self, weights_amount_in_neuron):
        for neuron in self.network_neurons:
            neuron.initialize_neuron_weights([random.uniform(self._PERCEPTRON_WEIGHT_MIN, self._PERCEPTRON_WEIGHT_MAX) for _ in range(0,
                                                                                                                                             weights_amount_in_neuron)])

    @classmethod
    def initialize_network_layer_with_random_weights(cls, weights_amount_in_neuron, neurons_layer):
        neurons_layer_with_weights = neurons_layer
        for neuron in neurons_layer_with_weights:
            neuron.initialize_neuron_weights([random.uniform(cls._PERCEPTRON_WEIGHT_MIN, cls._PERCEPTRON_WEIGHT_MAX) for _ in range(0,
                                                                                                                                           weights_amount_in_neuron)])
        return neurons_layer_with_weights

    @staticmethod
    def _init_network_with_neurons(neuron_amount_in_network):
        network_neurons = []
        for _ in xrange(neuron_amount_in_network):
            network_neurons.append(Neuron())
        return network_neurons

    def normalize_neural_network_neurons_weights(self, networks_neurons):
        neuron_array_weights_normalized = []
        for neuron in networks_neurons:
            neuron_with_modified_weights = Neuron()
            neuron_with_modified_weights.weights = self._return_normalized_vector(neuron.weights)
            neuron_array_weights_normalized.append(neuron_with_modified_weights)
        return neuron_array_weights_normalized

    def _return_normalized_vector(self, input_vector):
        flatten_vector = input_vector
        vector_length = numpy.linalg.norm(flatten_vector)
        vector_length_checked = lambda vector_length, flatten_vector: self.__compute_vector_length(flatten_vector) if vector_length == 0 else vector_length
        normalized_vector = []
        for vector_element in flatten_vector:
            normalized_vector.append(vector_element / vector_length_checked(vector_length, flatten_vector))
        return normalized_vector

    def __compute_vector_length(self, vector):
        sum = 0
        for vector_element in vector:
            sum += math.pow(vector_element, 2)
        result = numpy.sqrt(sum)
        return result

    def normalize_network_vectors(self):
        self.training_set = self._return_normalized_vector(self.training_set)

    def _compute_output_value_from_neuron(self, training_set, neuron_weights):
        result = 0
        for corresponding_index, training_sample in enumerate(training_set):
            result += training_sample * neuron_weights[corresponding_index]
        return result

    def compute_network_outputs(self):
        output_results = []
        for neuron in self.network_neurons:
            neuron_output = self._compute_output_value_from_neuron(self.training_set, neuron.weights)
            output_results.append(neuron_output)
        return output_results
