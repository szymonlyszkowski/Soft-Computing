import unittest
import numpy
from image_utils.image_scanner import ImageScanner


class ImageScannerTests(unittest.TestCase):
    def test_get_random_frame_length_from_image_8_by_8(self):
        # GIVEN
        mock_image_array = numpy.zeros((512, 512))
        scanner = ImageScanner()
        # WHEN
        random_frame = scanner.get_random_frame(mock_image_array, 8)
        # THEN
        self.assertEqual(len(random_frame), 64)

    def test_get_random_frame_length_from_image_4_by_4(self):
        # GIVEN
        mock_image_array = numpy.zeros((512, 512))
        scanner = ImageScanner()
        # WHEN
        random_frame = scanner.get_random_frame(mock_image_array, 4)
        # THEN
        self.assertEqual(len(random_frame), 16)

    def test_get_random_frame_length_from_image_16_by_16(self):
        # GIVEN
        mock_image_array = numpy.zeros((512, 512))
        scanner = ImageScanner()
        # WHEN
        random_frame = scanner.get_random_frame(mock_image_array, 16)
        # THEN
        self.assertEqual(len(random_frame), 256)


if __name__ == '__main__':
    unittest.main()
