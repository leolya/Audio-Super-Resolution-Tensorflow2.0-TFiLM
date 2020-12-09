import numpy as np
import librosa
from scipy.signal import hanning
import os
from tqdm import tqdm
import soundfile as sf
import tensorflow as tf


def SNR(y_true, y_pred):
    n_norm = np.mean((y_true - y_pred) ** 2)
    s_norm = np.mean(y_true ** 2)
    return 10 * np.log10((s_norm / n_norm) + 1e-8)


def get_power(x, nfft):
    S = librosa.stft(x, nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


def LSD(x_hr, x_pr):
    S1 = get_power(x_hr, nfft=2048)
    S2 = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd


def evaluation_whole(model, in_dir_hr, in_dir_lr, crop=32):

    snr_sum = 0
    lsd_sum = 0
    length_sum = 0

    hr_files = os.listdir(in_dir_hr)
    hr_files.sort()
    hr_file_list = []

    for hr_file in hr_files:
        hr_file_list.append(in_dir_hr + hr_file)

    lr_files = os.listdir(in_dir_lr)
    lr_files.sort()
    lr_file_list = []

    for lr_file in lr_files:
        lr_file_list.append(in_dir_lr + lr_file)

    file_num = len(hr_file_list)
    assert file_num == len(lr_file_list)

    for i in tqdm(range(file_num)):

        x_lr, fs = librosa.load(lr_file_list[i], sr=None)
        x_hr, fs_ = librosa.load(hr_file_list[i], sr=None)
        length = (len(x_lr) // crop) * crop
        x_lr = x_lr[0: length]
        x_hr = x_hr[0: length]

        x_in = np.expand_dims(np.expand_dims(np.abs(x_lr), axis=-1), axis=0)
        x_in = tf.convert_to_tensor(x_in, dtype=tf.float32)
        pred = model(x_in)
        pred = pred.numpy()
        pred = np.squeeze(np.squeeze(pred))

        snr = SNR(x_hr, pred)
        lsd = LSD(x_hr, pred)

        snr_sum = snr_sum + snr * length
        lsd_sum = lsd_sum + lsd * length

    return snr_sum / length_sum, lsd_sum / length_sum


def evaluation(model, crop_length, channel, in_dir_hr, in_dir_lr):

    snr_sum = 0
    lsd_sum = 0
    length_sum = 0

    window = hanning(crop_length)
    hr_files = os.listdir(in_dir_hr)
    hr_files.sort()
    hr_file_list = []

    for hr_file in hr_files:
        hr_file_list.append(in_dir_hr + hr_file)

    lr_files = os.listdir(in_dir_lr)
    lr_files.sort()
    lr_file_list = []

    for lr_file in lr_files:
        lr_file_list.append(in_dir_lr + lr_file)

    file_num = len(hr_file_list)
    assert file_num == len(lr_file_list)

    for i in tqdm(range(file_num)):
        if channel is None:
            x_lr, fs = librosa.load(lr_file_list[i], sr=None)
            x_hr, fs_ = librosa.load(hr_file_list[i], sr=None)
        else:
            x_lr, fs = librosa.load(lr_file_list[i], sr=None, mono=False)
            x_hr, fs_ = librosa.load(hr_file_list[i], sr=None, mono=False)
            x_lr = np.asfortranarray(x_lr[channel])
            x_hr = np.asfortranarray(x_hr[channel])

        length = len(x_hr)
        if length < crop_length:
            continue

        batches = int((length - crop_length / 2) / (crop_length / 2))
        x_hr = x_hr[0: int(batches * crop_length / 2 + crop_length / 2)]
        x_lr = x_lr[0: int(batches * crop_length / 2 + crop_length / 2)]
        length_sum = length_sum + length

        for j in range(0, batches):
            x_lr_ = x_lr[int(j * crop_length / 2): int((j * crop_length / 2) + crop_length)]
            x_in = np.expand_dims(np.expand_dims(np.abs(x_lr_), axis=-1), axis=0)
            x_in = tf.convert_to_tensor(x_in, dtype=tf.float32)
            pred = model(x_in)
            pred = pred.numpy()
            pred = np.squeeze(np.squeeze(pred))

            if j == 0:
                pred_audio_frame = pred * window
                pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
                pred_audio_end = pred_audio_frame[int(crop_length / 2):]

                pred_audio = pred[0: int(crop_length / 2)]
            else:
                pred_audio_frame = pred * window
                pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
                pred_overlap = pred_audio_font + pred_audio_end

                pred_audio = np.concatenate((pred_audio, pred_overlap), axis=0)

                pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            if j == batches - 1:
                pred_audio = np.concatenate((pred_audio, pred[int(crop_length / 2):]), axis=0)

        snr = SNR(x_hr, pred_audio)
        lsd = LSD(x_hr, pred_audio)

        snr_sum = snr_sum + snr * length
        lsd_sum = lsd_sum + lsd * length

    return snr_sum / length_sum, lsd_sum / length_sum


def generate_sr_sample(model, crop_length, in_dir_lr, save_path):

    window = hanning(crop_length)
    x_lr, fs = librosa.load(in_dir_lr, sr=None)
    length = x_lr.shape[0]
    batches = int((length - crop_length / 2) / (crop_length / 2))
    x_lr = x_lr[0: int(batches * crop_length / 2 + crop_length / 2)]

    for i in range(batches):
        x_lr_ = x_lr[int(i * crop_length / 2): int((i * crop_length / 2) + crop_length)]
        x_in = np.expand_dims(np.expand_dims(np.abs(x_lr_), axis=-1), axis=0)
        x_in = tf.convert_to_tensor(x_in, dtype=tf.float32)
        pred = model(x_in)
        pred = pred.numpy()
        pred = np.squeeze(np.squeeze(pred))

        if i == 0:
            pred_audio_frame = pred * window
            pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
            pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            pred_audio = pred[0: int(crop_length / 2)]
        else:
            pred_audio_frame = pred * window
            pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
            pred_overlap = pred_audio_font + pred_audio_end

            pred_audio = np.concatenate((pred_audio, pred_overlap), axis=0)

            pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            if i == batches - 1:
                pred_audio = np.concatenate((pred_audio, pred[int(crop_length / 2):]), axis=0)
    sf.write(save_path, pred_audio, samplerate=fs)
