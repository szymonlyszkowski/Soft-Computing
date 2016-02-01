import math


class ActivationFunction:
    def __init__(self):
        pass

    def identity_function(self, first_param):
        return first_param

    @classmethod
    def sigmoid_function(cls, neuron_output_sum):
        exponent = -1 * neuron_output_sum
        return 1 / (1 + math.exp(exponent))

    @classmethod
    def sigmoid_function_derivative(cls, argument):
        return ActivationFunction.sigmoid_function(argument) * (1 - ActivationFunction.sigmoid_function(argument))
