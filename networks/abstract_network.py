import random
from neurons.neuron import Neuron


class AbstractNeuralNetwork(object):
    def __init__(self, training_set):
        self.__training_set = training_set
        self.__network_neurons = self.__init_network_with_neurons(len(self.__training_set))
        self.__initialize_network_neurons_with_random_weights(self.__training_set)

    def apply_fixed_neurons_weights(self, two_dimensional_weights_array_to_be_set):
        self.apply_fixed_neurons_weights(self.__network_neurons, two_dimensional_weights_array_to_be_set)
        print 'Neuron weights set to: %s' % two_dimensional_weights_array_to_be_set

    def __initialize_network_neurons_with_random_weights(self, training_set):
        training_set_vector_size = len(training_set)
        for neuron in self.__network_neurons:
            neuron.initialize_neuron_weights([random.uniform(-1, 1) for _ in range(0, training_set_vector_size)])

    @staticmethod
    def __init_network_neurons_with_weights(network_neurons, two_dimensional_weights_array):
        vectors_of_weights_amount = len(two_dimensional_weights_array)
        neurons_amount_in_neural_network = len(network_neurons)
        if vectors_of_weights_amount != neurons_amount_in_neural_network:
            raise Exception('Amount of neurons: %d is not equal to vectors of weights amount: %d' % (neurons_amount_in_neural_network,
                                                                                                     vectors_of_weights_amount))
        for neuron_index, neuron in enumerate(network_neurons):
            neuron_recognizing_weights, neuron_identifier = two_dimensional_weights_array[neuron_index]
            neuron.initialize_neuron_weights(neuron_recognizing_weights)
            print 'Neuron of index %d recognizes %s' % (neuron_index, neuron_identifier)
        return network_neurons

    @staticmethod
    def __init_network_with_neurons(neuron_amount_in_network):
        network_neurons = []
        for _ in xrange(neuron_amount_in_network):
            network_neurons.append(Neuron())
        return network_neurons
