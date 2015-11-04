class Neuron:
    __SMALL_POSITIVE_REAL_NUMBER = 0.005

    def __init__(self):
        self.weights = list()
        self.activation_function = None
        self.training_set = list()

    def initialize_neuron(self, weights_list, training_samples_list):
        self.weights = weights_list
        self.training_set = training_samples_list

    def apply_training_set(self, training_set):
        self.training_set = training_set

    def apply_activation_function(self, activation_function):
        pass

    def compute_argument_value_for_activation_function(self):
        return sum(training_value*self.weights[corresponding_index] for corresponding_index, training_value in enumerate(self.training_set))

    def check_if_activate(self, argument):
        if argument > 0:
            return True
        return False

    def __delta_principle_for_identity_function(self, weight, desired_result, obtained_result, used_training_sample_for_iteration):
        result = weight + self.__SMALL_POSITIVE_REAL_NUMBER * (desired_result - obtained_result)*used_training_sample_for_iteration
        print 'result for delta principle %s '%result
        return result

    def apply_new_weights(self, desired_result, obtained_result):
        for corresponding_index,weight in enumerate(self.weights):
            used_training_sample_for_iteration = self.training_set.__getitem__(corresponding_index)
            print 'used training sample %s for index %d' % (used_training_sample_for_iteration, corresponding_index)
            print 'used weight %s' % weight
            new_weight = self.__delta_principle_for_identity_function(weight, desired_result, obtained_result, used_training_sample_for_iteration)
            self.weights.__setitem__(corresponding_index, new_weight)




