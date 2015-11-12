import random

import numpy

from neurons.neuron import Neuron


class MadalineNetwork:
    def __init__(self, training_set, neuron_amount_in_network, two_dimensional_array_weights):
        self.__training_set = training_set
        self.__madaline_network_neurons = self.__init_madaline_network_with_neurons(neuron_amount_in_network)
        self.__madaline_network_neurons = self.__init_madaline_network_neurons_with_weights(self.__madaline_network_neurons, two_dimensional_array_weights)

    @staticmethod
    def __initialize_madaline_network_neurons_with_random_weights(training_set):
        training_set_vector_size = len(training_set)
        for neuron in training_set:
            neuron.initialize_neuron_weights([random.uniform(-1, 1) for _ in range(0, training_set_vector_size)])

    @staticmethod
    def __init_madaline_network_neurons_with_weights(network_neurons, two_dimensional_weights_array):
        vectors_of_weights_amount = len(two_dimensional_weights_array)
        neurons_amount_in_neural_network = len(network_neurons)
        if vectors_of_weights_amount != neurons_amount_in_neural_network:
            raise Exception('Amount of neurons: %d is not equal to vectors of weights amount: %d' % (neurons_amount_in_neural_network,
                                                                                                     vectors_of_weights_amount))
        for neuron_index, neuron in enumerate(network_neurons):
            neuron.initialize_neuron_weights(two_dimensional_weights_array[neuron_index])
        return network_neurons

    @staticmethod
    def __init_madaline_network_with_neurons(neuron_amount_in_network):
        network_neurons = []
        for _ in xrange(neuron_amount_in_network):
            network_neurons.append(Neuron())
        print 'Network neurons created are: %s' % network_neurons
        return network_neurons

    def __return_normalized_vector(self, input_vector):
        flatten_vector = input_vector.flatten()
        vector_length = numpy.linalg.norm(flatten_vector)
        normalized_vector = []
        for vector_element in flatten_vector:
            normalized_vector.append(vector_element/vector_length)
        return normalized_vector

    def normalize_neural_network_neurons_weights(self, networks_neurons):
        neuron_array_weights_normalized = []
        for neuron in networks_neurons:
            neuron_with_modified_weights = Neuron()
            neuron_with_modified_weights.weights = self.__return_normalized_vector(neuron.weights)
            neuron_array_weights_normalized.append(neuron_with_modified_weights)
        return neuron_array_weights_normalized

    def run_madaline(self):
        self.__normalize_network_vectors()
        output_results = self.__compute_network_outputs()
        return self.__classify_output(output_results)

    def __compute_network_outputs(self):
        output_results = []
        for neuron in self.__madaline_network_neurons:
            neuron_output = self.__computue_output_value_from_neuron_in_madaline_network(self.__training_set, neuron.weights)
            output_results.append(neuron_output)
        return output_results

    def __classify_output(self, output_results):
        highest_output = max(output_results)
        highest_output_index = output_results.index(highest_output)
        print 'Neuron outputs: %s' % output_results
        print 'Neuron with highest output was: %s and its index is %s' % (highest_output, highest_output_index)
        return highest_output, highest_output_index

    def __normalize_network_vectors(self):
        self.__training_set = self.__return_normalized_vector(self.__training_set)
        self.__madaline_network_neurons = self.normalize_neural_network_neurons_weights(self.__madaline_network_neurons)

    def __computue_output_value_from_neuron_in_madaline_network(self, training_set, neuron_weights):
        result = 0
        for corresponding_index, training_sample in enumerate(training_set):
            result += training_sample*neuron_weights[corresponding_index]
        return result
