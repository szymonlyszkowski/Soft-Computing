import random
import numpy


class MadalineNetwork:
    def __init__(self, training_set, neuron_amount_in_network, neural_network_neuron):
        self.__training_set = training_set
        self.__madaline_network_neurons = self.__init_madaline_network_with_neurons(neuron_amount_in_network, neural_network_neuron)
        self.__init_madaline_network_neurons_with_weights(self.__madaline_network_neurons, neural_network_neuron)

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

    @staticmethod
    def __init_madaline_network_with_neurons(neuron_amount_in_network, neural_network_neuron):
        network_neurons = []
        for _ in xrange(neuron_amount_in_network):
            network_neurons.append(neural_network_neuron)
        print 'Network neurons created are: %s' % network_neurons
        return network_neurons


if __name__ == '__main__':
    print numpy.cos([1, 2, 3])
