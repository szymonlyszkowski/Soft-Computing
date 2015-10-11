__author__ = 'szymonidas'


class Neuron:

    __SMALL_POSITIVE_REAL_NUMBER = 1.5

    def __init__(self):
        self.weights = None
        self.activation_function = None
        self.training_set = None

    def apply_activation_function(self, activation_function):
        pass

    def __compute_new_values_in_training_set(self):
        modified_training_set = list()
        for training_sample in self.training_set:
            modified_training_set.append(self.__compute_new_training_sample(training_sample))
        return modified_training_set

    def __get_corresponding_weight(self, training_sample):
        return self.weights.__getitem__(self.training_set.index(training_sample))

    def __compute_new_training_sample(self, training_sample):
        return training_sample * self.__get_corresponding_weight(training_sample)

    def __compute_argument_value_for_activation_function(self):
        new_training_set_values = self.__compute_new_values_in_training_set()
        activation_function_argument = None
        for value in new_training_set_values:
            activation_function_argument +=value
        return activation_function_argument

    def check_if_activate(self, argument):
        if argument > 0:
            return True
        return False

    def delta_principle_for_identity_function(self, weight, desired_result, obtained_result):
        return weight + self.__SMALL_POSITIVE_REAL_NUMBER*(desired_result-obtained_result)

    def apply_new_weights(self,desired_result, obtained_result):
        for weight in self.training_set:
            self.delta_principle_for_identity_function(weight,desired_result,obtained_result)



