import numpy
from image_scanning.image_scanner import ImageScanner
from networks.abstract_network import AbstractNeuralNetwork


class KohonenNetwork(AbstractNeuralNetwork):
    __FRAME_SIZE = 4
    __WEIGHTS_AMOUNT_IN_NEURON = __FRAME_SIZE * __FRAME_SIZE
    __NEURONS_AMOUNT = 10

    def __init__(self, image_path):
        self.image_scanner = ImageScanner()
        self.image_array = self.image_scanner.get_image_as_array(image_path)
        self.neurons_used_indices = []
        super(KohonenNetwork, self).__init__(self.image_array.flatten(), self.__NEURONS_AMOUNT,
                                             self.__WEIGHTS_AMOUNT_IN_NEURON)

    def __prepare_kohonen_network(self):
        random_frame = self.image_scanner.get_random_frame(self.image_array, self.__FRAME_SIZE)
        self.training_set = random_frame
        self.normalize_network_vectors()

    def train_kohonen_network(self, training_periods):
        self.network_neurons = self.normalize_neural_network_neurons_weights(self.network_neurons)
        for _ in xrange(0, training_periods):
            self.__prepare_kohonen_network()
            neurons_outputs = self.compute_network_outputs()
            # maximum_value_on_output = numpy.nanmax(neurons_outputs)
            winner_neuron_index = numpy.nanargmax(neurons_outputs)
            # print "Output value of winning neuron: %s" % maximum_value_on_output
            # print "Neuron outputs: %s" % neurons_outputs
            # print "Index of winner Neuron: %d" % winner_neuron_index
            self.neurons_used_indices = self.__mark_neuron_as_used(self.neurons_used_indices, (winner_neuron_index, (numpy.mean(self.training_set))))
            self.network_neurons[winner_neuron_index].apply_new_weights_in_kohonnen_network(self.training_set)

    @classmethod
    def __mark_neuron_as_used(cls, neuron_indices_used, neuron_data):
        new_neuron_indices_used = neuron_indices_used
        if not neuron_data in new_neuron_indices_used:
            new_neuron_indices_used.append(neuron_data)
        return new_neuron_indices_used


def print_classifying_neurons_weights(indices_of_classifying_neurons, neurons):
    print "Neural network neurons classifying weights are:"
    for classifying_neuron_index, training_set_mean in indices_of_classifying_neurons:
        #print neurons[classifying_neuron_index].weights
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


if __name__ == '__main__':
    network = KohonenNetwork('../image_scanning/images/lena-512-grayscale.bmp')
    network.train_kohonen_network(100000)
    #print_classifying_neurons_weights(network.neurons_used_indices, network.network_neurons)
    #print_all_neurons_weights(network.network_neurons)
    indices = print_distinct_indices_values(network.neurons_used_indices)
    print "Indices of neurons used in network %s " % indices
    print "Neurons indices used for classification in Kohonen network frame mean values: %s" % network.neurons_used_indices