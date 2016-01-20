import random
import operator
from networks.abstract_network import AbstractNeuralNetwork
import teaching_patterns as patterns


class MultilayerPerceptron(AbstractNeuralNetwork):
    #output layer 4 neurons 2 weights each
    #hidden layer 2 neurons 4 weights each
    def __init__(self, training_set, neurons_amount_in_hidden_layer, weights_amount_in_hidden_layer, weights_amount_in_output_layer):
        self.training_set = training_set
        self.output_layer = self.__create_output_layer(4)
        self.output_layer = self.initialize_network_layer_with_random_weights(weights_amount_in_output_layer, random.randint(-0.5, 0.5), self.output_layer)
        super(MultilayerPerceptron, self).__init__(self, self.training_set, neurons_amount_in_hidden_layer, weights_amount_in_hidden_layer)

    def __create_output_layer(self, neurons_amount_in_output_layer):
        output_layer = self._init_network_with_neurons(neurons_amount_in_output_layer)
        return output_layer

    def __return_vector_from_neuron(self, training_set, neuron):
        result_vector = []
        for corresponding_index, training_sample in enumerate(training_set):
            result_vector.append(training_sample * neuron.neuron_weights[corresponding_index])
        return result_vector


    def __calculate_error_signal_for_hidden_layer(self, sigmoid_function_output, neuron):

        pass

    def __desired_vector_minus_vector_from_neuron(self, desired_vector, returned_vector):
        return map(operator.sub, desired_vector, returned_vector)


    def __calculate_error_signal_for_output_layer(self, sigmoid_function_output):
        pass






