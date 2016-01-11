import Image

import numpy
from image_utils.image_frame_slicer import ImageFrameSlicer
from image_utils.image_decoder import ImageDecoder
from image_utils.image_encoder import ImageEncoder
from networks.kohonen.kohonen_network import KohonenNetwork

if __name__ == '__main__':
    kohonen_network = KohonenNetwork('../image_utils/images/mandrill.png', 4, 50)
    kohonen_network.train_kohonen_network(10000)

    frame_slicer = ImageFrameSlicer(kohonen_network.image_array, kohonen_network.FRAME_SIZE)
    frames_array = frame_slicer.create_list_of_flatten_frames()

    image_encoder = ImageEncoder()
    encoded_data_array = image_encoder.encode_image(kohonen_network, frames_array)

    image_decoder = ImageDecoder(kohonen_network.FRAME_SIZE, frame_slicer.row_points, frame_slicer.column_points)
    decoded_image_array = image_decoder.decode_image(kohonen_network.image_array.shape, kohonen_network.network_neurons, encoded_data_array)
    img = Image.fromarray(decoded_image_array.astype(numpy.uint8), mode='L')
    img.show()
    image_name = '../image_utils/mandrill20K_%sx%s_frame_compression_%s_neurons.png' % (kohonen_network.FRAME_SIZE, kohonen_network.FRAME_SIZE,
                                                                                 kohonen_network.NEURONS_AMOUNT)
    img.save(image_name)
