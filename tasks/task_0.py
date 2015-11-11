from neurons.activation_function import ActivationFunction
from networks.simple_one_neuron_neural_network import SimpleOneNeuronNeuralNetwork
from neurons.neuron import Neuron

if __name__ == '__main__':
    network = SimpleOneNeuronNeuralNetwork()
    neuron = Neuron()
    activation_function = ActivationFunction()
    network.train_neural_network(neuron, activation_function)
    print "Neural network trained weights are: %s" % neuron.weights
