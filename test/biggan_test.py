import unittest
from scipy.stats import truncnorm
import numpy as np
from gantools import biggan
from gantools import image_utils

def create_random_input(dim_z, vocab_size, batch_size=1, truncation = 0.5, rand_seed = 123):
    def one_hot(index, dim):
        y = np.zeros((1,dim))
        if index < dim:
            y[0,index] = 1.0
        return y
    random_state = np.random.RandomState(rand_seed)
    vectors = truncation * truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=random_state)
    #TODO random labels
    labels = one_hot(0, vocab_size)#np.random.random_sample((vocab_size,))
    return vectors, labels, truncation

class TestBigGAN(unittest.TestCase):
    def test_biggan_sample(self):
        gan = biggan.BigGAN()
        vectors, labels, truncation = create_random_input(gan.dim_z, gan.vocab_size)
        ims = gan.sample(vectors, labels, truncation)
        image_utils.save_images(ims)


if __name__ == '__main__':
    unittest.main()
