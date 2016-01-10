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
                slice = (self.image_2d_array[start_row_point_slice:start_row_point_slice+self.frame_size,
                start_column_point_slice:start_column_point_slice+self.frame_size].flatten())
                slices.append(slice)
        return slices

    def decode_image(self,network_neurons, encoded_data_array):
        decoded_image = self.image_2d_array
        decoded_image.fill(0)
        frame_id = 0
        for start_row_point_slice in self. row_points:
            for start_column_point_slice in self.column_points:
                find_neuron_index_for_frame_id = lambda frame_id: [tuple_frame_id_neuron_index[0] for tuple_frame_id_neuron_index in encoded_data_array].index(
                        frame_id)
                tuple_index = find_neuron_index_for_frame_id(frame_id)
                neuron_index = encoded_data_array[tuple_index][1]
                decoding_neuron = network_neurons[neuron_index]
                decoded_frame = self.decode_pixel_values(decoding_neuron.weights)
                decoded_2d_frame = numpy.reshape(decoded_frame,(self.frame_size,self.frame_size))

                decoded_image[start_row_point_slice:start_row_point_slice+self.frame_size,
                start_column_point_slice:start_column_point_slice+self.frame_size] = decoded_2d_frame
                frame_id += 1
        return decoded_image

    def decode_pixel_values(self, neuron_weights):
        decoded_pixels = []
        for neuron_weight in neuron_weights:
            decoded_pixels.append(255 * neuron_weight)
        return decoded_pixels



if __name__ == '__main__':
    scanner = ImageScanner()
    image_array = scanner.get_image_as_array('images/lena-512-grayscale.bmp')
    frame_slicer = ImageFrameSlicer(image_array, 4)
    slices_array = frame_slicer.do_slices()
    numpy.savetxt('images/frames_array.txt', slices_array, delimiter=',', fmt="%s")
