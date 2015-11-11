import random

class SimpleOneNeuronNeuralNetwork:

    def __init__(self):
        self.expected_results = [19, 13, 13]
        self.whole_training_set = [[3, 2, 4], [2, 1, 3], [1, 3, 2]]
        self.initial_weights = ([random.uniform(0, 3) for _ in range(0, 3)])

    def train_neural_network(self, neuron, activation_function):
        neuron.initialize_neuron_weights(self.initial_weights)
        for teaching_period in xrange(100000):
            self.run_teaching_period(neuron, activation_function, self.whole_training_set)

    def run_teaching_period(self, neuron, activation_function, whole_training_set):
        for training_set_index, training_set in enumerate(whole_training_set):
            activation_function_argument = neuron.compute_argument_value_for_activation_function(training_set)
            if neuron.check_if_activate(activation_function_argument):
                activation_function.identity_function(activation_function_argument)
                neuron.apply_new_weights(self.expected_results[training_set_index], activation_function_argument, training_set)
            else:
                print 'Neuron not activated! Activation function argument is %s ' % activation_function_argument
