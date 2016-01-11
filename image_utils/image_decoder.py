import numpy


class ImageDecoder(object):
    def __init__(self, frame_size, row_points, column_points):
        self.frame_size = frame_size
        self.row_points = row_points
        self.column_points = column_points

    def __decode_2d_frame(self, decoding_neuron):
        decoded_frame = self.__decode_pixel_values(decoding_neuron.weights)
        decoded_2d_frame = numpy.reshape(decoded_frame, (self.frame_size, self.frame_size))
        return decoded_2d_frame

    @classmethod
    def __find_decoding_neuron_for_frame(cls, encoded_data_array, frame_id, network_neurons):
        find_neuron_index_for_frame_id = lambda frame_id: [tuple_frame_id_neuron_index[0] for tuple_frame_id_neuron_index in encoded_data_array].index(
            frame_id)
        tuple_index = find_neuron_index_for_frame_id(frame_id)
        neuron_index = encoded_data_array[tuple_index][1]
        decoding_neuron = network_neurons[neuron_index]
        return decoding_neuron

    def decode_image(self, image_shape, network_neurons, encoded_data_array):
        decoded_image = numpy.zeros(shape=image_shape)
        frame_id = 0
        for start_row_point_slice in self.row_points:
            for start_column_point_slice in self.column_points:
                decoding_neuron = self.__find_decoding_neuron_for_frame(encoded_data_array, frame_id, network_neurons)
                decoded_2d_frame = self.__decode_2d_frame(decoding_neuron)
                decoded_image[start_row_point_slice:start_row_point_slice + self.frame_size,
                start_column_point_slice:start_column_point_slice + self.frame_size] = decoded_2d_frame
                frame_id += 1
        return decoded_image

    @classmethod
    def __decode_pixel_values(cls, neuron_weights):
        decoded_pixels = []
        for neuron_weight in neuron_weights:
            decoded_pixels.append(255 * neuron_weight)
        return decoded_pixels