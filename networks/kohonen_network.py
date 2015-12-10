from image_scanning.image_scanner import ImageScanner
from networks.abstract_network import AbstractNeuralNetwork


class KohonenNetwork(AbstractNeuralNetwork):
    __FRAME_SIZE = 4
    __WEIGHTS_AMOUNT_IN_NEURON = __FRAME_SIZE * __FRAME_SIZE

    def __init__(self, image_path):
        self.image_scanner = ImageScanner()
        self.image_array = self.image_scanner.get_image_as_array(image_path)
        super(KohonenNetwork, self).__init__(self.image_array.flatten(), 20, self.__WEIGHTS_AMOUNT_IN_NEURON)

    def prepare_kohonen_network(self, image_path):
        pass

    def train_kohonen_network(self, training_periods):
        for _ in xrange(0, training_periods):
            random_frame = self.image_scanner.get_random_frame(self.image_array, self.__FRAME_SIZE)
            self.training_set = random_frame
            self.normalize_neural_network_neurons_weights(self.network_neurons)
            self.normalize_network_vectors()
            outputs = self.compute_network_outputs()
            print outputs.index(max(outputs))





if __name__ == '__main__':
    network = KohonenNetwork('../image_scanning/images/lena-512-grayscale.bmp')
    network.train_kohonen_network(10000)
    print "Neural network trained weights are: %s"
