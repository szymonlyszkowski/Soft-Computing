from PIL import Image
import numpy
from image_utils.image_frame_slicer import ImageFrameSlicer
from image_utils.image_scanner import ImageScanner
from networks.abstract_network import AbstractNeuralNetwork


class KohonenNetwork(AbstractNeuralNetwork):
    _FRAME_SIZE = 4
    __WEIGHTS_AMOUNT_IN_NEURON = _FRAME_SIZE * _FRAME_SIZE
    __NEURONS_AMOUNT = 200

    def __init__(self, image_path):
        self.image_scanner = ImageScanner()
        self.image_array = self.image_scanner.get_image_as_array(image_path)
        self.neurons_used_indices = []
        super(KohonenNetwork, self).__init__(self.image_array.flatten(), self.__NEURONS_AMOUNT,
                                             self.__WEIGHTS_AMOUNT_IN_NEURON)

    def __prepare_kohonen_network(self):
        random_frame = self.image_scanner.get_random_frame(self.image_array, self._FRAME_SIZE)
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

    def encode_image(self, frames_array):
        encoded = []
        frame_id = 0
        for frame in frames_array:
            self.training_set = frame
            neurons_outputs = self.compute_network_outputs()
            winner_neuron_index = numpy.nanargmax(neurons_outputs)
            # frame_mean = self.training_set.mean()
            # print 'Neuron of index %d for frame mean %f encoded frame %s' % (winner_neuron_index, frame_mean, frame)
            encoded.append((frame_id, winner_neuron_index))
            frame_id += 1
        return encoded

    def decode_image(self, frames_array_encoded, frame_size):

        pass


def print_classifying_neurons_weights(indices_of_classifying_neurons, neurons):
    print "Neural network neurons classifying weights are:"
    for classifying_neuron_index, training_set_mean in indices_of_classifying_neurons:
        # print neurons[classifying_neuron_index].weights
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
    network = KohonenNetwork('../image_utils/images/lena-512-grayscale.bmp')
    network.train_kohonen_network(20000)

    frame_slicer = ImageFrameSlicer(network.image_array, network._FRAME_SIZE)
    frames_array = frame_slicer.do_slices()

    encoded_data_array = network.encode_image(frames_array)
    print encoded_data_array

    decoded_image_array = frame_slicer.decode_image(network.network_neurons, encoded_data_array)
    print decoded_image_array

    img = Image.fromarray(decoded_image_array)
    img.show()
    img.save('test.bmp')
    numpy.savetxt('decoded_image_array.txt', decoded_image_array, delimiter=',', fmt="%s")
    decoded=numpy.asarray(decoded_image_array,dtype=numpy.uint8) #if values still in range 0-255!

    decoded_image=Image.fromarray(decoded,mode='L')
    decoded_image.save('out.jpg')




    # print_classifying_neurons_weights(network.neurons_used_indices, network.network_neurons)
    # print_all_neurons_weights(network.network_neurons)
    indices = print_distinct_indices_values(network.neurons_used_indices)
    print "Indices of neurons used in network %s " % indices
    # print "Neurons indices used for classification in Kohonen network frame mean values: %s" % network.neurons_used_indices

    # RAMKA o ID=xyz ZAKODOWANA PRZEZ NEURON 1 pusty obrazek wymnazam wagi neuronu 1 (skaluje przez 255) i odzysuje ramke
