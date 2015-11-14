import numpy


class Letters:
    def __init__(self):
        pass

    def get_T_four_by_four(self):
        t_sample_array = numpy.array([[1, 1, 1, 1],
                                      [0, 1, 1, 0],
                                      [0, 1, 1, 0],
                                      [0, 1, 1, 0]])
        return t_sample_array, 'Letter T'

    def get_V_four_by_four(self):
        v_sample_array = numpy.array([[1, 0, 0, 1],
                                      [1, 0, 0, 1],
                                      [0, 1, 1, 0],
                                      [0, 1, 1, 0]])
        return v_sample_array, 'Letter V'

    def get_W_four_by_four(self):
        w_sample_array = numpy.array([[1, 0, 0, 1],
                                      [1, 0, 0, 1],
                                      [1, 1, 1, 1],
                                      [1, 0, 0, 1]])
        return w_sample_array, 'Letter W'

    def get_X_four_by_four(self):
        x_sample_array = numpy.array([[1, 0, 0, 1],
                                      [0, 1, 1, 0],
                                      [0, 1, 1, 0],
                                      [1, 0, 0, 1]])
        return x_sample_array, 'Letter X'

    def get_Y_four_by_four(self):
        y_sample_array = numpy.array([[1, 0, 0, 1],
                                      [0, 1, 1, 0],
                                      [0, 1, 0, 0],
                                      [1, 0, 0, 0]])
        return y_sample_array, 'Letter Y'

    def get_Z_four_by_four(self):
        z_sample_array = numpy.array([[1, 1, 1, 1],
                                      [0, 0, 1, 0],
                                      [0, 1, 0, 0],
                                      [1, 1, 1, 1]])
        return z_sample_array, 'Letter Z'

    def get_all_letters(self):
        return [self.get_T_four_by_four(), self.get_V_four_by_four(), self.get_W_four_by_four(), self.get_X_four_by_four(), self.get_Y_four_by_four(),
                self.get_Z_four_by_four()]
