import Image
import numpy
from image_utils.image_scanner import ImageScanner

if __name__ == '__main__':

    scanner = ImageScanner()
    image_array = scanner.get_image_as_array('../image_utils/images/lena-512-grayscale.bmp')
    print image_array.size
    numpy.savetxt('normal_image_array.txt', image_array, delimiter=',', fmt="%s")
    data =  numpy.loadtxt('result.txt',delimiter=',', dtype=numpy.uint8)
    data = data
    print data.size


    print "Image array shape %s data shape %s" % (image_array.shape, data.shape)
    image = Image.fromarray(numpy.array(data))
    print image
    image.show()
