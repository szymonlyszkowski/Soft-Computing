from networks.multilayer_perceptron.multilayer_perceptron import MultilayerPerceptron
from networks.multilayer_perceptron.teaching_patterns import teaching_patterns_with_desired_outputs

if __name__ == '__main__':
    patterns = teaching_patterns_with_desired_outputs()
    perceptron = MultilayerPerceptron(patterns[0][0], 2, 4, 4, 2)

    #PROPAGATE AND ADJUST WEIGHTS
    hidden_layer_outputs = perceptron.compute_network_outputs()
    error_signals_output_layer = perceptron.calculate_error_signals_for_output_layer_neurons(patterns[0][1])
    error_signals_hidden_layer = perceptron.calculate_error_signals_for_hidden_layer(error_signals_output_layer)
    perceptron.apply_new_weights_in_layer(perceptron.output_layer, error_signals_output_layer, hidden_layer_outputs)
    perceptron.apply_new_weights_in_layer(perceptron.network_neurons, error_signals_hidden_layer, patterns[0][1])

    #CHECK RESULTS
    hidden_layer_out = perceptron.compute_network_outputs()
    output_layer_out = perceptron.compute_weighted_sums_from_output_layer_neurons(hidden_layer_out)
    print output_layer_out
    print hidden_layer_out
