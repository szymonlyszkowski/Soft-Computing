from ActivationFunction import ActivationFunction
from SimpleOneNeuronNeuralNetwork import SimpleOneNeuronNeuralNetwork
from Neuron import Neuron

if __name__ == '__main__':
    network = SimpleOneNeuronNeuralNetwork()
    neuron = Neuron()
    activation_function = ActivationFunction()
    network.train_neural_network(neuron, activation_function)
    print "Neural network trained weights are: %s" % neuron.weights
