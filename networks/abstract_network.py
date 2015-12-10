import random
import numpy
from neurons.neuron import Neuron


class AbstractNeuralNetwork(object):
    def __init__(self, training_set, neurons_amount, weights_amount_in_neuron):
        self.training_set = training_set
        self.network_neurons = self.__init_network_with_neurons(neurons_amount)
        self.__initialize_network_neurons_with_random_weights(weights_amount_in_neuron)

    def __initialize_network_neurons_with_random_weights(self, weights_amount_in_neuron):
        for neuron in self.network_neurons:
            neuron.initialize_neuron_weights([random.uniform(-1, 1) for _ in range(0, weights_amount_in_neuron)])

    @staticmethod
    def __init_network_with_neurons(neuron_amount_in_network):
        network_neurons = []
        for _ in xrange(neuron_amount_in_network):
            network_neurons.append(Neuron())
        return network_neurons

    def normalize_neural_network_neurons_weights(self, networks_neurons):
        neuron_array_weights_normalized = []
        for neuron in networks_neurons:
            neuron_with_modified_weights = Neuron()
            neuron_with_modified_weights.weights = self.__return_normalized_vector(neuron.weights)
            neuron_array_weights_normalized.append(neuron_with_modified_weights)
        return neuron_array_weights_normalized

    def __return_normalized_vector(self, input_vector):
        flatten_vector = input_vector
        vector_length = numpy.linalg.norm(flatten_vector)
        normalized_vector = []
        for vector_element in flatten_vector:
            normalized_vector.append(vector_element / vector_length)
        return normalized_vector

    def normalize_network_vectors(self):
        self.training_set = self.__return_normalized_vector(self.training_set)

    def __computue_output_value_from_neuron(self, training_set, neuron_weights):
        result = 0
        for corresponding_index, training_sample in enumerate(training_set):
            result += training_sample * neuron_weights[corresponding_index]
        return result

    def compute_network_outputs(self):
        output_results = []
        for neuron in self.network_neurons:
            neuron_output = self.__computue_output_value_from_neuron(self.training_set, neuron.weights)
            output_results.append(neuron_output)
        return output_results