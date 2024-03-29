import numpy
from image_utils.image_scanner import ImageScanner
from networks.abstract_network import AbstractNeuralNetwork


class KohonenNetwork(AbstractNeuralNetwork):
    def __init__(self, image_path, frame_size, neurons_amount):
        self.FRAME_SIZE = frame_size
        self.NEURONS_AMOUNT = neurons_amount
        self.__WEIGHTS_AMOUNT_IN_NEURON = frame_size * frame_size
        self.image_scanner = ImageScanner()
        self.image_array = self.image_scanner.get_image_as_array(image_path)
        self.neurons_used_indices = []
        super(KohonenNetwork, self).__init__(self.image_array.flatten(), self.NEURONS_AMOUNT,
                                             self.__WEIGHTS_AMOUNT_IN_NEURON)

    def __prepare_kohonen_network(self):
        random_frame = self.image_scanner.get_random_frame(self.image_array, self.FRAME_SIZE)
        self.training_set = random_frame
        self.normalize_network_vectors()

    def train_kohonen_network(self, training_periods):
        self.network_neurons = self.normalize_neural_network_neurons_weights(self.network_neurons)
        for _ in xrange(0, training_periods):
            self.__prepare_kohonen_network()
            neurons_outputs = self.compute_network_outputs()
            winner_neuron_index = numpy.nanargmax(neurons_outputs)
            self.neurons_used_indices = self.__mark_neuron_as_used(self.neurons_used_indices, (winner_neuron_index, (numpy.mean(self.training_set))))
            winner_neuron = self.network_neurons[winner_neuron_index]
            winner_neuron.apply_new_weights_in_kohonnen_network(self.training_set)
            winner_neuron.weights = self._return_normalized_vector(winner_neuron.weights)

    @classmethod
    def __mark_neuron_as_used(cls, neuron_indices_used, neuron_data):
        new_neuron_indices_used = neuron_indices_used
        if not neuron_data in new_neuron_indices_used:
            new_neuron_indices_used.append(neuron_data)
        return new_neuron_indices_used


def print_classifying_neurons_weights(indices_of_classifying_neurons, neurons):
    print "Neural network neurons classifying weights are:"
    for classifying_neuron_index, training_set_mean in indices_of_classifying_neurons:
        print "Neuron %s of index %d classifies frame with mean %s \n" % (neurons[classifying_neuron_index], classifying_neuron_index, training_set_mean)


def print_all_neurons_weights(neurons):
    print "Neural network all neurons trained weights are:"
    for neuron in neurons:
        print neuron.weights


def print_distinct_indices_values(neurons_used_indices):
    indices_set = set()
    for tuple_index_mean in neurons_used_indices:
        indices_set.add(tuple_index_mean[0])
    return indices_set
