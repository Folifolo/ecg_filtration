import pickle as pkl
from math import sqrt
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import cwt, ricker, periodogram

DEFAULT_NOISE_PROB = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]

em_path = 'em' + "_500Hz.pkl"
ma_path = 'ma' + "_500Hz.pkl"
bw_path = 'bw' + "_500Hz.pkl"

with open(em_path, 'rb') as f:
    em = pkl.load(f)
with open(ma_path, 'rb') as f:
    ma = pkl.load(f)
with open(bw_path, 'rb') as f:
    bw = pkl.load(f)


def _generator_carcass(ecgs, size, batch_size, noise_type='ma', level=1):
    x_batch = np.zeros((batch_size, size))
    y_batch = np.zeros((batch_size, size))
    for i in range(batch_size):
        ecg_number = randint(0, ecgs.shape[0] - 1)
        start_position = randint(0, ecgs.shape[1] - size - 1)

        ecg_sample = ecgs[ecg_number, start_position:start_position + size]
        noised_sample = _add_noise(ecg_sample, noise_type, level)

        x_batch[i] = ecg_sample
        y_batch[i] = noised_sample

    return x_batch, y_batch


def noised_for_filtration(ecgs, size, batch_size, lead=0, noise_type='ma', level=1):
    while 1:
        (x_batch, y_batch) = _generator_carcass(ecgs[:, :, lead], size, batch_size, noise_type, level)
        x_batch = np.expand_dims(x_batch, 2)
        y_batch = np.expand_dims(y_batch, 2)
        x_batch = np.expand_dims(x_batch, 3)
        y_batch = np.expand_dims(y_batch, 3)

        yield (x_batch, y_batch)


def noised_for_detection(ecgs, size, batch_size, lead=0, noise_type='ma', level=1):
    while 1:
        (x_batch, y_batch) = _generator_carcass(ecgs[:, :, lead], size, batch_size, noise_type, level)
        for i in range(batch_size):
            mask = _generate_noised_fragments_position2(size)
            x_batch[i] = x_batch[i] * (1 - mask) + y_batch[i] * mask
            y_batch[i] = mask

        x_batch = np.expand_dims(x_batch, 2)
        y_batch = np.expand_dims(y_batch, 2)

        yield (x_batch, y_batch)


def noised_spectrogram_for_detection(ecgs, size, batch_size, noise_prob, lead=0, noise_type='ma', level=1):
    while 1:
        widths = np.arange(1, 51)
        spectr_batch = np.zeros((batch_size, 50, 200, 1))
        labels_batch = np.zeros((batch_size, 2))
        (x_batch, y_batch) = _generator_carcass(ecgs[:, :, lead], size, batch_size, noise_type, level)

        for i in range(batch_size):
            if np.random.random() < noise_prob:
                cwtmatr = cwt(y_batch[i], ricker, widths)
                labels_batch[i, 1] = 1
            else:
                cwtmatr = cwt(x_batch[i], ricker, widths)
                labels_batch[i, 0] = 1

            spectr_batch[i, :, :, 0] = cwtmatr[:, ::25]

        yield (spectr_batch, labels_batch)


def noised_for_classification(ecgs, size, batch_size, noise_prob, lead=0, noise_type='ma', level=1):
    while 1:
        ecg_batch = np.zeros((batch_size, size))
        labels_batch = np.zeros((batch_size, 2))
        (x_batch, y_batch) = _generator_carcass(ecgs[:, :, lead], size, batch_size, noise_type, level)

        for i in range(batch_size):
            if np.random.random() < noise_prob:
                ecg_batch[i] = y_batch[i]
                labels_batch[i, 1] = 1
            else:
                ecg_batch[i] = x_batch[i]
                labels_batch[i, 0] = 1

        ecg_batch = np.expand_dims(ecg_batch, 2)

        yield (ecg_batch, labels_batch)


def _generate_segment_artefact(ecg, type=0, noise_type='ma'):
    if type == 0:
        return ecg
    elif type < 5:
        return _add_noise(ecg, noise_type, type)
    elif type == 5:
        noise_sample, _ = _get_noise_snr(noise_type, 1)
        noise_start_position = randint(0, noise_sample.shape[0] - len(ecg) - 1)
        noise_channel = randint(0, 1)
        noise_fragment = noise_sample[noise_start_position:noise_start_position + len(ecg), noise_channel]
        _, noise_power = periodogram(noise_fragment, 500)
        while np.mean(noise_power) == 0:
            noise_start_position = randint(0, noise_sample.shape[0] - len(ecg) - 1)
            noise_channel = randint(0, 1)
            noise_fragment = noise_sample[noise_start_position:noise_start_position + len(ecg), noise_channel]
            _, noise_power = periodogram(noise_fragment, 500)

        _, ecg_power = periodogram(ecg, 500)

        noise_fragment = noise_fragment * sqrt(np.mean(ecg_power)) / sqrt(np.mean(noise_power))

        noise_fragment = noise_fragment + np.mean(ecg) - np.mean(noise_fragment)
        size = 30
        tmp_left = np.arange(0, 1, 1 / size)
        tmp_right = np.arange(1, 0, -1 / size)
        noise_fragment[:size] = ecg[:size] * tmp_right + noise_fragment[:size] * tmp_left
        noise_fragment[-size:] = ecg[-size:] * tmp_left + noise_fragment[-size:] * tmp_right

        return noise_fragment


def autocorrelator(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def artefact_for_detection(ecg, size, batch_size, noise_prob=None, noise_type='ma'):
    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB
    while 1:
        ecg_batch = np.zeros((batch_size, size, 1))
        mask_batch = np.zeros((batch_size, size, 6))
        mask_batch[:, :, 0] = np.ones((batch_size, size))
        for i in range(batch_size):
            x_tmp = choose_ecg_fragment(ecg, size)
            mask_tmp = np.zeros((size, 6))
            mask_tmp[:, 0] = np.ones((size))
            intervals = create_mask(size)
            for j in range(intervals.shape[0] - 1):
                type = np.random.choice(6, p=noise_prob)
                fragment = _generate_segment_artefact(x_tmp[intervals[j]:intervals[j + 1]], type, noise_type)
                mask_tmp[intervals[j]:intervals[j + 1], 0] = 0
                mask_tmp[intervals[j]:intervals[j + 1], type] = 1

                x_tmp[intervals[j]:intervals[j + 1]] = fragment

            ecg_batch[i, :, 0] = x_tmp
            mask_batch[i] = mask_tmp

        yield (ecg_batch, mask_batch)


def artefact_for_detection_3_in_2_out(ecg, size, batch_size, noise_prob=None, noise_type='ma', num_sections=0):
    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB
    denum = 16
    while 1:
        ecg_batch = np.zeros((batch_size, size, 1))
        ecg_batch1 = np.zeros((batch_size, size // denum, 1))
        ecg_batch2 = np.zeros((batch_size, 500, 1))
        mask_batch = np.zeros((batch_size, size, 6))
        mask_batch[:, :, 0] = np.ones((batch_size, size))
        mask_batch1 = np.zeros((batch_size, 2))
        for i in range(batch_size):
            x_tmp = choose_ecg_fragment(ecg, size)
            mask_tmp = np.zeros((size, 6))
            mask_tmp[:, 0] = np.ones(size)
            intervals = create_mask(size, num_sections)
            for j in range(intervals.shape[0] - 1):
                type = np.random.choice(6, p=noise_prob)
                fragment = _generate_segment_artefact(x_tmp[intervals[j]:intervals[j + 1]], type, noise_type)
                mask_tmp[intervals[j]:intervals[j + 1], 0] = 0
                mask_tmp[intervals[j]:intervals[j + 1], type] = 1

                x_tmp[intervals[j]:intervals[j + 1]] = fragment

            ecg_batch[i, :, 0] = x_tmp
            ecg_batch1[i, :, 0] = x_tmp[::denum]
            ecg_batch2[i, :, 0] = autocorrelator(x_tmp)[:500]
            mask_batch[i] = mask_tmp
            mask_batch1[i, 1] = np.sum(mask_tmp[:, 4:]) / mask_tmp.shape[0]
            mask_batch1[i, 0] = 1 - mask_batch1[i, 1]
        yield ([ecg_batch, ecg_batch1, ecg_batch2], [mask_batch, mask_batch1])


def artefact_for_detection_dual(ecg, size, batch_size, noise_prob=None, noise_type='ma'):
    if noise_prob is None:
        noise_prob = DEFAULT_NOISE_PROB
    while 1:
        ecg_batch = np.zeros((batch_size, size, 1))
        ecg_batch1 = np.zeros((batch_size, size // 8, 1))
        mask_batch = np.zeros((batch_size, size, 6))
        mask_batch[:, :, 0] = np.ones((batch_size, size))
        for i in range(batch_size):
            x_tmp = choose_ecg_fragment(ecg, size)

            mask_tmp = np.zeros((size, 6))
            mask_tmp[:, 0] = np.ones(size)
            intervals = create_mask(size)
            for j in range(intervals.shape[0] - 1):
                type = np.random.choice(6, p=noise_prob)
                fragment = _generate_segment_artefact(x_tmp[intervals[j]:intervals[j + 1]], type, noise_type)
                mask_tmp[intervals[j]:intervals[j + 1], 0] = 0
                mask_tmp[intervals[j]:intervals[j + 1], type] = 1

                x_tmp[intervals[j]:intervals[j + 1]] = fragment

            ecg_batch[i, :, 0] = x_tmp
            ecg_batch1[i, :, 0] = x_tmp[::8]
            mask_batch[i] = mask_tmp
        yield ([ecg_batch, ecg_batch1], mask_batch)


def choose_ecg_fragment(ecg, size):
    if ecg.ndim == 1:
        start_position = randint(0, ecg.shape[0] - size - 1)
        ret = np.copy(ecg[start_position:start_position + size])
    elif ecg.ndim == 3:
        ecg_num = randint(0, ecg.shape[0] - 1)
        start_position = randint(0, ecg.shape[1] - size - 1)
        ecg_lead = randint(0, ecg.shape[2] - 1)
        ret = np.copy(ecg[ecg_num, start_position:start_position + size, ecg_lead])
    else:
        raise ValueError('Wrong ecg dimensionality')
    return ret


def create_mask(size, num_sections=10, min_size=300):
    if num_sections != 0:
        intervals = np.random.randint(size, size=num_sections)
        intervals = np.append(intervals, [0, size])
    else:
        intervals = np.array([0, size])
    intervals = np.sort(intervals)
    difference = np.diff(intervals)
    counter = 0
    for j in range(len(difference)):
        if difference[j] < min_size:
            intervals = np.delete(intervals, j + 1 - counter)
            counter += 1
    return intervals


def _add_noise(ecg, noise_type='em', level=1):
    size = len(ecg)
    noise_sample, snr = _get_noise_snr(noise_type, level)

    noise_start_position = randint(0, noise_sample.shape[0] - size - 1)
    noise_channel = randint(0, 1)
    noise_fragment = noise_sample[noise_start_position:noise_start_position + size, noise_channel]

    noise_power = np.sum(noise_fragment ** 2, axis=0)
    ecg_power = np.sum(ecg ** 2, axis=0)

    target_noise = ecg_power / (10 ** (snr / 10))

    noise_fragment = noise_fragment * sqrt(target_noise) / sqrt(np.mean(noise_power))

    return ecg + noise_fragment


def _get_noise_snr(noise_type, level):
    if noise_type == 'bw':
        noise_sample = bw
        if level == 1:
            snr = 12
        elif level == 2:
            snr = 6
        elif level == 3:
            snr = 0
        elif level == 4:
            snr = -6
        else:
            raise ValueError('This noise level is not supported')
    elif noise_type == 'em':
        noise_sample = em
        if level == 1:
            snr = 6
        elif level == 2:
            snr = 0
        elif level == 3:
            snr = -6
        elif level == 4:
            snr = -12
        else:
            raise ValueError('This noise level is not supported')
    elif noise_type == 'ma':
        noise_sample = ma
        if level == 1:
            snr = 12
        elif level == 2:
            snr = 6
        elif level == 3:
            snr = 0
        elif level == 4:
            snr = -6
        else:
            raise ValueError('This noise level is not supported')
    else:
        raise ValueError('This noise type is not supported')

    return noise_sample, snr


def _generate_mask(ecg_size):
    mask = np.zeros(ecg_size)
    boundaries = []
    right = 0
    while right < ecg_size - 1:
        rnd = np.random.random()
        if rnd < 0.8 * ((ecg_size - right) / ecg_size):
            segment_size = np.random.randint(1, ecg_size - right)
            segment_start = np.random.randint(right, ecg_size - segment_size)
            right = segment_start + segment_size
            boundaries.append([segment_start, right])
        else:
            right = ecg_size
    for bound in boundaries:
        mask[bound[0]:bound[1]] = 1
    return mask


def _generate_noised_fragments_position2(ecg_size):
    mask = np.zeros(ecg_size)
    boundaries = [0, ecg_size]
    max_block_size = ecg_size
    min_block_size = ecg_size
    max_pos = 0
    while min_block_size > ecg_size / 20:
        rnd = np.random.randint(0, max_block_size)
        boundaries.append(rnd + max_pos)
        boundaries.sort()
        min_block_size = np.min(np.diff(np.array(boundaries)))
        max_block_size = np.max(np.diff(np.array(boundaries)))
        max_pos = np.argmax(np.diff(np.array(boundaries)))

    boundaries.sort()
    plt.scatter(boundaries, [0] * len(boundaries))
    for i in range(len(boundaries) - 1):
        rnd = np.random.random()
        if rnd < 1 - ((boundaries[i + 1] - boundaries[i]) / ecg_size):
            mask[boundaries[i]:boundaries[i + 1]] = 1

    return mask


def _resize_signal(noise, old_freq=360, new_freq=500):
    len_seconds = (noise.shape[0] // old_freq)
    noise = noise[:len_seconds * old_freq]
    x = np.arange(0, len_seconds, 1 / old_freq)

    tck = interpolate.splrep(x, noise, s=0)
    xnew = np.arange(0, len_seconds, 1 / new_freq)
    ynew = interpolate.splev(xnew, tck, der=0)

    return ynew


def resize_noise(noise, name, old_freq=360, new_freq=500):
    len_seconds = (noise.shape[0] // old_freq)
    new_noise = np.zeros((len_seconds * new_freq, noise.shape[1]))
    for i in range(noise.shape[1]):
        new_noise[:, i] = _resize_signal(noise[:, i], old_freq, new_freq)

    with open(name + '.pkl', 'wb') as output:
        pkl.dump(new_noise, output)


if __name__ == "__main__":
    import scipy.io as sio
    import glob
    import wfdb
    lst = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116,
           117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210,
           212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    res = []
    for elem in lst:
        rec = np.zeros((902500, 2))
        record = wfdb.rdsamp('mit\\' + str(elem))
        record = record[0]
        rec[:,0] = _resize_signal(record[:,0])
        rec[:,1] = _resize_signal(record[:,1])
        res.append(np.array(rec))
    res = np.array(res)
    with open("mit\\mit_dataset.pkl", 'wb') as output:
        pkl.dump(res, output)

