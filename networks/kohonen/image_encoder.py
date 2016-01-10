import numpy


class ImageEncoder(object):
    def __init__(self):
        pass

    def encode_image(self, kohonen_network, frames_array):
        """
        Returns array of tuples consisting: frame_id and index of neuron decoding frame with given id.
        """
        encoded = []
        frame_id = 0
        for frame in frames_array:
            kohonen_network.training_set = frame
            neurons_outputs = kohonen_network.compute_network_outputs()
            winner_neuron_index = numpy.nanargmax(neurons_outputs)
            encoded.append((frame_id, winner_neuron_index))
            frame_id += 1
        return encoded