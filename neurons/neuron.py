class Neuron:
    __TEACHING_FACTOR = 0.05

    def __init__(self):
        self.weights = []

    def initialize_neuron_weights(self, weights_list):
        self.weights = weights_list

    def compute_argument_value_for_activation_function(self, training_set):
        return sum(training_value*self.weights[corresponding_index] for corresponding_index, training_value in enumerate(training_set))

    def __delta_principle_for_identity_function(self, weight, desired_result, obtained_result, used_training_sample_for_iteration):
        result = weight + self.__TEACHING_FACTOR * (desired_result - obtained_result)*used_training_sample_for_iteration
        return result

    def apply_new_weights(self, desired_result, obtained_result, training_set):
        for corresponding_index, weight in enumerate(self.weights):
            used_training_sample_for_iteration = training_set.__getitem__(corresponding_index)
            new_weight = self.__delta_principle_for_identity_function(weight, desired_result, obtained_result, used_training_sample_for_iteration)
            self.weights.__setitem__(corresponding_index, new_weight)

    def apply_new_weights_in_kohonnen_network(self, training_set):
        for corresponding_index, weight in enumerate(self.weights):
            used_training_sample_for_iteration = training_set.__getitem__(corresponding_index)
            new_weight = self.__delta_principle_for_identity_function(weight, used_training_sample_for_iteration, weight,1)
            self.weights.__setitem__(corresponding_index, new_weight)



def check_if_activate(argument):
    if argument > 0:
        return True
    return False




