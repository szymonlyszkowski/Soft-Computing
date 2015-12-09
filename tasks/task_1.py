import numpy

from networks.madeline_network import MadalineNetwork
from training_set_letters import letters

if __name__ == '__main__':
    training_set = numpy.array([[1, 1, 1, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [1, 1, 1, 1]])
    letter = letters.Letters()
    MadalineNetwork(training_set, letter.get_all_letters()).run_madaline()
