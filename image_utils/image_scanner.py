from PIL import Image
import random
import numpy


class ImageScanner(object):
    def __init__(self):
        pass

    @staticmethod
    def get_image_as_array(image_path):
        image = Image.open(image_path)
        return numpy.array(image)

    def get_random_frame(self, image_as_array, frame_side_length):
        maximum_allowed_index = self.__get_maxium_allowed_index(image_as_array, frame_side_length)
        end_index, start_index = self.__get_random_allowed_indices(frame_side_length, maximum_allowed_index)
        one_dimensional_array = []
        for row in xrange(start_index, end_index):
            for column in xrange(start_index, end_index):
                one_dimensional_array.append(image_as_array[row][column])
        return one_dimensional_array

    @staticmethod
    def __get_random_allowed_indices(frame_side_length, maximum_allowed_index):
        start_index = random.randint(0, maximum_allowed_index)
        end_index = start_index + frame_side_length
        return end_index, start_index

    @staticmethod
    def __compute_frame_value(frame_array):
        return numpy.mean(frame_array)

    @staticmethod
    def __get_maxium_allowed_index(image_array, frame_side_length):
        """
        Returns maxium value for index to be used for getting random frame for size == frame_side_length
        from image_array
        :type frame_side_length: integer
        :type image_array: 1d or 2d array
        """
        maximum_allowed_index = len(image_array) - frame_side_length
        return maximum_allowed_index


if __name__ == '__main__':
    scanner = ImageScanner()
    image_array = scanner.get_image_as_array('images/lena-512-grayscale.bmp')
    numpy.savetxt('images/image_array.txt', image_array, delimiter=',', fmt="%s")
