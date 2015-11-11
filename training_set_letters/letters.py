import numpy
class Letters:
    def __init__(self):
        pass

    def get_X_four_by_four(self):
        x_sample_array = numpy.array([[1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1]])
        return x_sample_array

    def get_Y_four_by_four(self):
        y_sample_array = numpy.array([[1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])
        return y_sample_array

    def get_Z_four_by_four(self):
        z_sample_array = numpy.array([[1, 1, 1, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 1, 1, 1]])
        return z_sample_array
    # numpy.linalg.norm(z_sample_array) = uzysknie normy wektora w przypadku z = sqrt(10), potem kazdy element wektora dzielimy przez jego norme
