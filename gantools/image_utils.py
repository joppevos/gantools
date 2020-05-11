import os
import PIL.Image

def save_image(arr, fp, format='JPEG'):
    image = PIL.Image.fromarray(arr)
    image.save(fp, format=format, quality=90)

def save_images(ims, output_dir='', prefix='', format='JPEG'):
    for i, im in enumerate(ims):
        full_path = os.path.join(output_dir, prefix + str(i).zfill(4) + '.' + format.lower())
        save_image(im, full_path, format)

class ImageSaver(object):
    def __init__(self, output_dir='', prefix='', image_format='JPEG'):
        self.output_dir = str(output_dir)
        self.prefix = str(prefix)
        self.image_format = str(image_format)
        self.index = int(0)
        self.quality = 90

    def save(self, ims):
        for i, im in enumerate(ims):
            full_path = os.path.join(
                    self.output_dir,
                    self.prefix + str(self.index).zfill(4) + '.' + self.image_format.lower()
                    )
            image = PIL.Image.fromarray(im)
            image.save(full_path, format=self.image_format, quality=self.quality)
            self.index += 1

