import random
from networks.multilayer_perceptron.multilayer_perceptron import MultilayerPerceptron
from networks.multilayer_perceptron.teaching_patterns import teaching_patterns_with_desired_outputs


def print_weights(layer, layer_name):
    print "Layer %s contain following weights:" % layer_name
    for neuron in layer:
        print neuron.weights


def train_network(training_periods_amount, list_of_tuples_training_set_expected_outputs, perceptron, shuffle=True):
    list_of_tuples_training_set_expected_outputs_to_use = list_of_tuples_training_set_expected_outputs
    for _ in xrange(training_periods_amount):
        run_training_period(list_of_tuples_training_set_expected_outputs_to_use,perceptron)
        if shuffle:
            random.shuffle(list_of_tuples_training_set_expected_outputs_to_use)


def run_training_period(list_of_tuples_training_set_expected_outputs, perceptron):
    for pattern in list_of_tuples_training_set_expected_outputs:
        perceptron.training_set = pattern[0]
        hidden_layer_outputs = perceptron.compute_network_outputs()
        error_signals_output_layer = perceptron.calculate_error_signals_for_output_layer_neurons(pattern[1])
        error_signals_hidden_layer = perceptron.calculate_error_signals_for_hidden_layer(error_signals_output_layer)
        perceptron.apply_new_weights_in_layer(perceptron.output_layer, error_signals_output_layer, hidden_layer_outputs)
        perceptron.apply_new_weights_in_layer(perceptron.network_neurons, error_signals_hidden_layer, pattern[0])


if __name__ == '__main__':
    patterns = teaching_patterns_with_desired_outputs()
    perceptron = MultilayerPerceptron(None, 2, 4, 4, 2)
    perceptron.add_bias()
    train_network(100000, patterns, perceptron)

    # CHECK RESULTS
    perceptron.training_set = patterns[0][0]
    print "EXPECTED PATTERN: %s" % patterns[0][1]
    hidden_layer_out = perceptron.compute_network_outputs()
    output_layer_out = perceptron.compute_output_layer_outputs(hidden_layer_out)
    print "OBTAINED PATTERN: %s" % output_layer_out

    perceptron.training_set = patterns[1][0]
    print "EXPECTED PATTERN: %s" % patterns[1][1]
    hidden_layer_out = perceptron.compute_network_outputs()
    output_layer_out = perceptron.compute_output_layer_outputs(hidden_layer_out)
    print "OBTAINED PATTERN: %s" % output_layer_out

    perceptron.training_set = patterns[2][0]
    print "EXPECTED PATTERN: %s" % patterns[2][1]
    hidden_layer_out = perceptron.compute_network_outputs()
    output_layer_out = perceptron.compute_output_layer_outputs(hidden_layer_out)
    print "OBTAINED PATTERN: %s" % output_layer_out

    perceptron.training_set = patterns[3][0]
    print "EXPECTED PATTERN: %s" % patterns[3][1]
    hidden_layer_out = perceptron.compute_network_outputs()
    output_layer_out = perceptron.compute_output_layer_outputs(hidden_layer_out)
    print "OBTAINED PATTERN: %s" % output_layer_out
