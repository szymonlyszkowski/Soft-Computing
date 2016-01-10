import numpy
from image_utils.image_scanner import ImageScanner


class ImageFrameSlicer(object):
    def __init__(self, image_2d_array, frame_size):
        self.image_2d_array = image_2d_array
        self.frame_size = frame_size
        self.column_points = self._calculate_column_iteration_points()
        self.row_points = self._calculate_row_iteration_points()

    def _calculate_row_iteration_points(self):
        row_length = self.image_2d_array.shape[0]
        row_elements = range(0, row_length)
        return row_elements[0::self.frame_size]

    def _calculate_column_iteration_points(self):
        column_length = self.image_2d_array.shape[1]
        row_elements = range(0, column_length)
        return row_elements[0::self.frame_size]

    def do_slices(self):
        slices = []
        for start_row_point_slice in self. row_points:
            for start_column_point_slice in self.column_points:
                slices.append(self.image_2d_array[start_row_point_slice:start_row_point_slice+self.frame_size,
                start_column_point_slice:start_column_point_slice+self.frame_size].flatten())
        return slices




if __name__ == '__main__':
    scanner = ImageScanner()
    image_array = scanner.get_image_as_array('images/lena-512-grayscale.bmp')
    frame_slicer = ImageFrameSlicer(image_array, 2)
    slices_array = frame_slicer.do_slices()
    print slices_array
    numpy.savetxt('images/frames_array.txt', slices_array, delimiter=',', fmt="%s")
