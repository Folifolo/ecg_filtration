import glob
import pickle as pkl
import random
from math import sqrt
from random import randint
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import cwt, ricker, periodogram

from dataset import load_dataset

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

        noised_sample = _add_noise(ecgs, noise_type, level)

        x_batch[i] = ecg_sample
        y_batch[i] = noised_sample

    return x_batch, y_batch


def noised_for_filtration(ecgs, size, batch_size, lead=0, noise_type='ma', level=1):
    while 1:
        (x_batch, y_batch) = _generator_carcass(ecgs[:, :, lead], size, batch_size, noise_type, level)
        x_batch = np.expand_dims(x_batch, 2)
        y_batch = np.expand_dims(y_batch, 2)

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
        _, ecg_power = periodogram(ecg, 500)

        noise_fragment = noise_fragment * sqrt(np.mean(ecg_power)) / sqrt(np.mean(noise_power))

        return noise_fragment


def artefact_for_detection_holter(ecg, size, batch_size, noise_prob, noise_type='ma'):
    while 1:
        x_batch = np.zeros((batch_size, size, 1))
        y_batch = np.zeros((batch_size, size, 2))
        for i in range(batch_size):
            start_position = randint(0, ecg.shape[0] - size - 1)
            x_tmp = ecg[start_position:start_position + size]
            mask_tmp = np.zeros((size, 2))

            intervals = np.random.randint(size, size=10)
            intervals = np.append(intervals, [0, size])
            intervals = np.sort(np.unique(intervals))
            for j in range(intervals.shape[0]-1):
                type = random.choice([0,5])
                fragment = _generate_segment_artefact(ecg[intervals[j]:intervals[j+1]], type, noise_type)
                if type == 0:
                    mask_tmp[intervals[j]:intervals[j+1], 0] = 1
                elif type == 5:
                    mask_tmp[intervals[j]:intervals[j+1], 1] = 1

                x_tmp[intervals[j]:intervals[j+1]] = fragment



            x_batch[i, :, 0] = x_tmp
            y_batch[i] = mask_tmp

        yield (x_batch, y_batch)


def _add_noise(ecg, noise_type='em', level=1):
    size = len(ecg)

    if noise_type not in ['em', 'bw', 'ma']:
        raise ValueError('This noise type is not supported')
    if level not in [1, 2, 3, 4]:
        raise ValueError('This noise level is not supported')

    noise_sample, snr = _get_noise_snr(noise_type, level)

    noise_start_position = randint(0, noise_sample.shape[0] - size - 1)
    noise_channel = randint(0, 1)
    noise_fragment = noise_sample[noise_start_position:noise_start_position + size, noise_channel]

    _, noise_power = periodogram(noise_fragment, 500)
    _, ecg_power = periodogram(ecg, 500)

    target_noise = (10 ** (-snr / 10)) * np.mean(ecg_power)

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
    with open('holters\\holter0.pkl', 'rb') as infile:
        dataset = pkl.load(infile)
    gener = artefact_for_detection_holter(dataset, 15000, 1, 0, noise_type='ma')
    next(gener)
    xy = load_dataset()
    X = xy["x"]
    #_generate_segment_artefact(X[0, :, 0], type =5)
    #_add_noise(X[0, :, 0])
