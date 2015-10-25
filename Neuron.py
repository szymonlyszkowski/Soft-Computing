import ActivationFunction

__author__ = 'szymonidas'


class Neuron:
    __SMALL_POSITIVE_REAL_NUMBER = 0.05

    def __init__(self):
        self.weights = list()
        self.activation_function = None
        self.training_set = list()

    def initialize_neuron(self, weights_list, training_samples_list):
        self.weights = weights_list
        self.training_set = training_samples_list

    def apply_activation_function(self, activation_function):
        pass

    def __get_corresponding_weight(self, training_sample):
        return self.weights.__getitem__(self.training_set.index(training_sample))

    def __compute_new_training_sample(self, training_sample):
        return training_sample * self.__get_corresponding_weight(training_sample)

    def compute_argument_value_for_activation_function(self):
        activation_function_argument = 0
        for training_sample_value in self.training_set:
            new_value = training_sample_value * self.weights.__getitem__(self.training_set.index(training_sample_value))
            print 'new value %s' % new_value
            activation_function_argument += new_value
        return activation_function_argument

    def check_if_activate(self, argument):
        if argument > 0:
            return True
        return False

    def __delta_principle_for_identity_function(self, weight, desired_result, obtained_result, used_training_sample_for_iteration):
        result = weight + self.__SMALL_POSITIVE_REAL_NUMBER * (desired_result - obtained_result)*used_training_sample_for_iteration
        print 'result for delta principle %s '%result
        return result

    def apply_new_weights(self, desired_result, obtained_result):
        for weight in self.weights:
            computed_index = self.weights.index(weight)
            used_training_sample_for_iteration = self.training_set.__getitem__(computed_index)
            print 'used training sample for iteration %s ' % used_training_sample_for_iteration
            print 'index %s '% computed_index
            new_weight = self.__delta_principle_for_identity_function(weight, desired_result, obtained_result, used_training_sample_for_iteration)
            print 'new weight %s' % new_weight
            self.weights.__setitem__(computed_index, new_weight)




