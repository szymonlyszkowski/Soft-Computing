from PIL import Image
import random
import numpy


def get_maxium_allowed_index(image_array, frame_side_length):
    """
    Returns maxium value for index to be used for getting random frame for size == frame_side_length
    from image_array
    :type frame_side_length: integer
    :type image_array: 1d or 2d array
    """
    maximum_allowed_index = len(image_array) - frame_side_length
    return maximum_allowed_index


class ImageScanner(object):

    def __init__(self, image_path):
        self.image_path = image_path

    def get_image_as_array(self):
        image = Image.open(self.image_path)
        return numpy.array(image)



    def __scan_image(self, image_array, frame_side_length):
        pass

    def __get_random_frame(self, image_array, maximum_allowed_index, frame_side_length):
        end_index_row, start_index_row = self.__get_random_allowed_indices(frame_side_length, maximum_allowed_index)
        end_index_column, start_index_column = self.__get_random_allowed_indices(frame_side_length, maximum_allowed_index)
        one_dimensional_array = []
        for row in xrange(start_index_row,end_index_row):
            for column in xrange(start_index_column, end_index_column):
                one_dimensional_array.append(image_array[row][column])
        return one_dimensional_array

    def __get_random_allowed_indices(self, frame_side_length, maximum_allowed_index):
        start_index = random.uniform(0, maximum_allowed_index)
        end_index = start_index + frame_side_length
        return end_index, start_index

    def __compute_frame_value(self, frame_array):
        return numpy.mean(frame_array)



if __name__ == '__main__':
    scanner = ImageScanner('images/lena-512-grayscale.bmp')
    image_array = scanner.get_image_as_array()
    print len(image_array)
    numpy.savetxt('images/image_array.txt', image_array, delimiter=',', fmt="%s")
