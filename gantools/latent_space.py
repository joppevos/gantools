import numpy as np
from scipy import signal, interpolate

def one_hot(index, dim):
    y = np.zeros((1, dim))
    if index < dim:
        y[0, index] = 1.0
    return y

# interpolation methods
def linear_interp(points, step_count):
    def linear_interp1d(y):
        x = np.linspace(0., 1., len(y))
        xnew = np.linspace(0., 1., step_count)
        return interpolate.interp1d(x, y)(xnew)
    return np.apply_along_axis(linear_interp1d, 0, points)

def cubic_spline_interp(points, step_count):
    def cubic_spline_interp1d(y):
        x = np.linspace(0., 1., len(y))
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.linspace(0., 1., step_count)
        return interpolate.splev(xnew, tck, der=0)
    if points.shape[0] < 4:
        raise ValueError('Too few points for cubic interpolation: need 4, got {}'.format(points.shape[0]))
    return np.apply_along_axis(cubic_spline_interp1d, 0, points)


# TODO: the math in this function is embarrasingly bad. fix at some point.
def sequence_keyframes(keyframes, num_frames, batch_size=1, interp_method='linear', loop=False):
    interp_fn = {
            'linear': linear_interp,
            'cubic': cubic_spline_interp,
            }[interp_method]
    div = int(num_frames // len(keyframes))
    rem = int(num_frames - (div * len(keyframes)))
    frame_counts = np.full((len(keyframes), ), div) + \
            np.append(np.ones((rem,), dtype=int), np.zeros((len(keyframes) - rem, ), dtype=int))
    batch_div = int(num_frames // batch_size)
    batch_rem = 1 if int(num_frames % batch_size) > 0 else 0
    batch_count = batch_div + batch_rem

    if loop is True:
        keyframes.append(keyframes[0])# seq returns to start

    truncation_keys = np.asarray([keyframe['truncation'] for keyframe in keyframes])
    z_keys = np.asarray([np.asarray(keyframe['latent']) * keyframe['truncation'] for keyframe in keyframes])
    label_keys = np.asarray([keyframe['label'] for keyframe in keyframes])

    z_seq = interp_fn(z_keys, num_frames)
    label_seq = interp_fn(label_keys, num_frames)
    truncation_seq = interp_fn(truncation_keys, num_frames)

    # you can only change trunc once per batch
    truncation_seq_resampled = np.full((1),truncation_seq[0])\
            if batch_count is 1\
            else signal.resample(truncation_seq, batch_count)
    return np.asarray(z_seq), np.asarray(label_seq), truncation_seq_resampled
