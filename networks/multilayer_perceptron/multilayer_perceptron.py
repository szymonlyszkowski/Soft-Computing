import random
import numpy
from networks.abstract_network import AbstractNeuralNetwork
from neurons.activation_function import ActivationFunction


class MultilayerPerceptron(AbstractNeuralNetwork):
    # output layer 4 neurons 2 weights each
    # hidden layer 2 neurons 4 weights each
    def __init__(self, training_set, neurons_amount_in_hidden_layer, neurons_amount_in_output_layer, weights_amount_in_hidden_layer_neuron,
                 weights_amount_in_output_layer_neuron):
        self.training_set = training_set
        self.output_layer = self.__create_output_layer(neurons_amount_in_output_layer)
        self.output_layer = self.initialize_network_layer_with_random_weights(weights_amount_in_output_layer_neuron,
                                                                              self.output_layer)
        super(MultilayerPerceptron, self).__init__(self.training_set, neurons_amount_in_hidden_layer, weights_amount_in_hidden_layer_neuron)

    def __create_output_layer(self, neurons_amount_in_output_layer):
        output_layer = self._init_network_with_neurons(neurons_amount_in_output_layer)
        return output_layer

    def calculate_error_signals_for_hidden_layer(self, output_layer_error_signals):
        error_signals = []
        hidden_layer_outputs, output_layer_weighted_sum_of_errors_and_neurons_weights = self.__prepare_data_to_calculate_hidden_layer_errors(
            output_layer_error_signals)
        for corresponding_index, neuron in enumerate(self.network_neurons):
            error_signals.append(self.__calculate_error_signal_for_hidden_layer(corresponding_index, hidden_layer_outputs,
                                                                                output_layer_weighted_sum_of_errors_and_neurons_weights))
        return error_signals

    def __calculate_error_signal_for_hidden_layer(self, corresponding_index, hidden_layer_outputs, output_layer_weighted_sum_of_errors_and_neurons_weights):
        derivative = ActivationFunction.sigmoid_function_derivative(hidden_layer_outputs[corresponding_index])
        weighted_sum = output_layer_weighted_sum_of_errors_and_neurons_weights[corresponding_index]
        error_signal = derivative * weighted_sum
        return error_signal

    def __prepare_data_to_calculate_hidden_layer_errors(self, output_layer_error_signals):
        hidden_layer_outputs = self.compute_network_outputs()
        output_layer_weighted_sums_of_errors_and_neurons_weights = []
        for hidden_layer_neuron_index, hidden_layer_output in enumerate(hidden_layer_outputs):
            output_layer_weighted_sums_of_errors_and_neurons_weights.append(self.__calculate_weighted_sum_of_output_layer_error_and_its_weights(
                hidden_layer_neuron_index, output_layer_error_signals))
        return hidden_layer_outputs, output_layer_weighted_sums_of_errors_and_neurons_weights

    def __calculate_weighted_sum_of_output_layer_error_and_its_weights(self, hidden_layer_neuron_index, output_layer_errors):
        weighted_sum = 0
        for corresponding_index, neuron in enumerate(self.output_layer):
            weighted_sum += neuron.weights[hidden_layer_neuron_index] * output_layer_errors[corresponding_index]
        return weighted_sum

    def calculate_error_signals_for_output_layer_neurons(self, expected_vector):
        error_signals = []
        weighted_sums_from_hidden_layer = self.compute_network_outputs()
        obtained_outputs_from_output_layer_neurons = self.__compute_output_layer_outputs(weighted_sums_from_hidden_layer)
        weighted_sums_from_output_layer = self.compute_weighted_sums_from_output_layer_neurons(weighted_sums_from_hidden_layer)
        for corresponding_index, neuron in enumerate(self.output_layer):
            neuron_weighted_sum = weighted_sums_from_output_layer[corresponding_index]
            obtained_output_from_output_layer_neuron = obtained_outputs_from_output_layer_neurons[corresponding_index]
            expected_output_from_output_layer_neuron = expected_vector[corresponding_index]
            error_signal = self.__calculate_error_signal_for_output_layer(expected_output_from_output_layer_neuron, obtained_output_from_output_layer_neuron,
                                                                          neuron_weighted_sum)
            error_signals.append(error_signal)
        return error_signals

    def __calculate_error_signal_for_output_layer(self, expected_vector_element, obtained_vector_element_from_output_layer_neuron, neuron_weighted_sum):
        derivative = ActivationFunction.sigmoid_function_derivative(neuron_weighted_sum)
        result = expected_vector_element - obtained_vector_element_from_output_layer_neuron
        error_signal_vector = numpy.multiply(result, derivative)
        return error_signal_vector

    def compute_weighted_sums_from_output_layer_neurons(self, output_values_from_hidden_layer):
        # self.network_neurons stands for hidden layer!!!
        neurons_weighted_sums = []
        neuron_weighted_sum = 0
        for neuron in self.output_layer:
            for corresponding_index, hidden_layer_output in enumerate(output_values_from_hidden_layer):
                neuron_weighted_sum += hidden_layer_output * neuron.weights[corresponding_index]
            neurons_weighted_sums.append(neuron_weighted_sum)
            neuron_weighted_sum = 0
        return neurons_weighted_sums

    def __compute_output_layer_outputs(self, training_set):
        outputs = []
        for neuron in self.output_layer:
            outputs.append(self.__return_vector_from_neuron_after_multiplication_with_training_set(training_set, neuron))
        return outputs

    @staticmethod
    def __return_vector_from_neuron_after_multiplication_with_training_set(training_set, neuron):
        result_vector = 0
        for corresponding_index, neuron_weight in enumerate(neuron.weights):
            training_sample = training_set[corresponding_index]
            result_vector += training_sample * neuron_weight
        return result_vector

    def apply_new_weights_in_layer(self, layer, error_values, training_set):
        for index, neuron in enumerate(layer):
            training_sample = neuron.compute_argument_value_for_activation_function(training_set)
            new_weights = self.__compute_new_weights_for_neuron(neuron, error_values[index], training_sample)
            neuron.weights = new_weights

    def __compute_new_weights_for_neuron(self, neuron, error_value, training_sample):
        new_weights = []
        for weight in neuron.weights:
            new_weight = (weight + 0.005 * error_value * training_sample)
            new_weights.append(new_weight)
        return new_weights