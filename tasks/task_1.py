import numpy

from networks.madeline_network import MadalineNetwork
from training_set_letters import letters

if __name__ == '__main__':
    training_set = numpy.array([[1, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 1]])
    letter = letters.Letters()
    MadalineNetwork(training_set,3, [letter.get_X_four_by_four(), letter.get_Y_four_by_four(), letter.get_Z_four_by_four()]).run_madaline()