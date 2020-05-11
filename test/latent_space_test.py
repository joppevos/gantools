import unittest
import numpy as np
from scipy.stats import truncnorm
import context
from gantools import latent_space
import PIL.Image
import math

def create_random_keyframe(n_vector, n_label):
    truncation = (0.9 - 0.1)*np.random.random() + 0.1
    random_state = np.random.RandomState()
    vectors = truncnorm.rvs(-2, 2, size=(n_vector,), random_state=random_state)
    keyframe = {
            'vector': vectors.tolist(),
            'label': latent_space.one_hot(np.random.randint(0, n_label), n_label),
            'truncation': truncation,
            }
    return keyframe

#### TMP
def save_image(arr, fp):
    image = PIL.Image.fromarray(arr)
    image.save(fp, format='JPEG', quality=90)

def save_ims(ims):
    i = 0
    for im in ims:
        path = './GAN_'+str(i).zfill(3)+'.jpeg'
        save_image(im, path)
        i += 1
####

def compare_float_arrays_2d(target_seq, actual_seq):
    for target, actual in zip(target_seq, actual_seq):
        for ti, ai in zip(target, actual):
            assert math.isclose(ti, ai), 'target: %s; actual %s' % (str(target), str(actual))

class TestLatentSpace(unittest.TestCase):
    def test_linear_interp_basic(self):
        target_seq = np.asarray([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]).transpose()
        points = np.asarray([target_seq[0], target_seq[-1]])
        step_count = target_seq.shape[0]
        actual_seq = latent_space.linear_interp(points, step_count)
        compare_float_arrays_2d(target_seq, actual_seq)

    def test_sequence_keyframes_linear_random_basic(self):
        n_keyframes = 10
        n_vector = 100
        n_label = 1000
        num_frames = 100
        keyframes = np.asarray([create_random_keyframe(n_vector, n_label) for i in range(n_keyframes)])
        z, label, trunc = latent_space.sequence_keyframes(
                keyframes,
                num_frames,
                batch_size=1,
                interp_method='linear')

        assert (num_frames == z.shape[0]),\
                'z sequence: target frame count: %s; actual shape: %s' % (num_frames, z.shape)
        assert (num_frames == label.shape[0]),\
                'label sequence: target frame count: %s; actual shape: %s' % (num_frames, label.shape)
        assert (num_frames == trunc.shape[0]),\
                'trunc sequence: target frame count: %s; actual shape: %s' % (num_frames, trunc.shape)

    def test_sequence_keyframes_linear_random_batch(self):
        n_keyframes = 10
        n_vector = 100
        n_label = 1000
        num_frames = 100
        batch_size = 7 # pick something that doesn't divide num_frames
        batch_div = int(num_frames // batch_size)
        batch_rem = 1 if int(num_frames % batch_size) > 0 else 0
        batch_count = batch_div + batch_rem
        keyframes = np.asarray([create_random_keyframe(n_vector, n_label) for i in range(n_keyframes)])
        z, label, trunc = latent_space.sequence_keyframes(
                keyframes,
                num_frames,
                batch_size=batch_size,
                interp_method='linear')

        assert (num_frames == z.shape[0]),\
                'z sequence: target frame count: %s; actual shape: %s' % (num_frames, z.shape)
        assert (num_frames == label.shape[0]),\
                'label sequence: target frame count: %s; actual shape: %s' % (num_frames, label.shape)
        assert (batch_count == trunc.shape[0]),\
                'trunc sequence: target frame count: %s; actual shape: %s' % (batch_count, trunc.shape)

    def test_sequence_keyframes_linear_random_batch_oob(self):
        n_keyframes = 10
        n_vector = 100
        n_label = 1000
        num_frames = 100
        batch_size = 150 # pick something that doesn't divide num_frames
        batch_div = int(num_frames // batch_size)
        batch_rem = 1 if int(num_frames % batch_size) > 0 else 0
        batch_count = batch_div + batch_rem
        keyframes = np.asarray([create_random_keyframe(n_vector, n_label) for i in range(n_keyframes)])
        z, label, trunc = latent_space.sequence_keyframes(
                keyframes,
                num_frames,
                batch_size=batch_size,
                interp_method='linear')

        assert (num_frames == z.shape[0]),\
                'z sequence: target frame count: %s; actual shape: %s' % (num_frames, z.shape)
        assert (num_frames == label.shape[0]),\
                'label sequence: target frame count: %s; actual shape: %s' % (num_frames, label.shape)
        assert (batch_count == trunc.shape[0]),\
                'trunc sequence: target frame count: %s; actual shape: %s' % (batch_count, trunc.shape)

    def test_sequence_keyframes_cubic_random_basic(self):
        n_keyframes = 10
        n_vector = 100
        n_label = 1000
        num_frames = 100
        keyframes = np.asarray([create_random_keyframe(n_vector, n_label) for i in range(n_keyframes)])
        z, label, trunc = latent_space.sequence_keyframes(
                keyframes,
                num_frames,
                batch_size=1,
                interp_method='cubic')

        assert (num_frames == z.shape[0]),\
                'z sequence: target frame count: %s; actual shape: %s' % (num_frames, z.shape)
        assert (num_frames == label.shape[0]),\
                'label sequence: target frame count: %s; actual shape: %s' % (num_frames, label.shape)
        assert (num_frames == trunc.shape[0]),\
                'trunc sequence: target frame count: %s; actual shape: %s' % (num_frames, trunc.shape)

    def test_sequence_keyframes_cubic_random_batch(self):
        n_keyframes = 10
        n_vector = 100
        n_label = 1000
        num_frames = 100
        batch_size = 7 # pick something that doesn't divide num_frames
        batch_div = int(num_frames // batch_size)
        batch_rem = 1 if int(num_frames % batch_size) > 0 else 0
        batch_count = batch_div + batch_rem
        keyframes = np.asarray([create_random_keyframe(n_vector, n_label) for i in range(n_keyframes)])
        z, label, trunc = latent_space.sequence_keyframes(
                keyframes,
                num_frames,
                batch_size=batch_size,
                interp_method='cubic')

        assert (num_frames == z.shape[0]),\
                'z sequence: target frame count: %s; actual shape: %s' % (num_frames, z.shape)
        assert (num_frames == label.shape[0]),\
                'label sequence: target frame count: %s; actual shape: %s' % (num_frames, label.shape)
        assert (batch_count == trunc.shape[0]),\
                'trunc sequence: target frame count: %s; actual shape: %s' % (batch_count, trunc.shape)

    def test_sequence_keyframes_cubic_random_batch_oob(self):
        n_keyframes = 10
        n_vector = 100
        n_label = 1000
        num_frames = 100
        batch_size = 150 # pick something that doesn't divide num_frames
        batch_div = int(num_frames // batch_size)
        batch_rem = 1 if int(num_frames % batch_size) > 0 else 0
        batch_count = batch_div + batch_rem
        keyframes = np.asarray([create_random_keyframe(n_vector, n_label) for i in range(n_keyframes)])
        z, label, trunc = latent_space.sequence_keyframes(
                keyframes,
                num_frames,
                batch_size=batch_size,
                interp_method='cubic')

    def test_circle(self):
        n_vector = 100
        n_label = 1000
        center = create_random_keyframe(n_vector, n_label)['vector']
        normal = create_random_keyframe(n_vector, n_label)['vector']
        latent_space.circle([center, normal], step_count)


if __name__ == '__main__':
    unittest.main()
