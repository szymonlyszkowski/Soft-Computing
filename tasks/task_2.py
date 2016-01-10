import Image
from image_utils.image_frame_slicer import ImageFrameSlicer
from networks.kohonen.image_decoder import ImageDecoder
from networks.kohonen.image_encoder import ImageEncoder
from networks.kohonen.kohonen_network import KohonenNetwork


if __name__ == '__main__':

    kohonen_network = KohonenNetwork('../image_utils/images/lena-512-grayscale.bmp')
    kohonen_network.train_kohonen_network(20000)

    frame_slicer = ImageFrameSlicer(kohonen_network.image_array, kohonen_network._FRAME_SIZE)
    frames_array = frame_slicer.create_list_of_flatten_frames()

    image_encoder = ImageEncoder()
    encoded_data_array = image_encoder.encode_image(kohonen_network, frames_array)

    image_decoder = ImageDecoder(kohonen_network._FRAME_SIZE,frame_slicer.row_points,frame_slicer.column_points)
    decoded_image_array = image_decoder.decode_image(kohonen_network.image_array.shape, kohonen_network.network_neurons, encoded_data_array)

    img = Image.fromarray(decoded_image_array)
    img.show()
    img.save('../image_utils/kohonen_output_image.png')

