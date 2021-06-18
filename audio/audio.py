import librosa
import os
import re
import soundfile
import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.io import wavfile


class Audio:
    def __init__(self, audio_hparams):
        self.hps = audio_hparams

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hps.sample_rate)[0]

    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.hps.sample_rate, wav.astype(np.int16))
        return

    def spectrogram(self, y, clip_norm=True):
        D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.hps.ref_level_db
        if clip_norm:
            S = self._normalize(S)
        return S

    def logf0(self, wav_path, lower_f0, upper_f0):
        sr = self.hps.sample_rate
        hop_len = self.hps.frame_shift_sample
        save_path = wav_path.split('/')[-1].split('.')[0] + '.lf0'
        temp_raw_path = wav_path.split('/')[-1].split('.')[0] + '.short'
        fs_khz = sr / 1000
        os.system('sox {} -t raw {}'.format(wav_path, temp_raw_path))
        os.system('x2x +sf {} | pitch -H {} -L {} -p {} -s {} -o 2 > {}'.format(
            temp_raw_path, upper_f0, lower_f0, hop_len, fs_khz, save_path))
        lf0s = np.fromfile(save_path, dtype=np.float32)
        os.system('rm {}'.format(temp_raw_path))
        os.system('rm {}'.format(save_path))
        return lf0s

    def inv_spectrogram(self, spectrogram):
        S = self._db_to_amp(self._denormalize(spectrogram) + self.hps.ref_level_db)
        return self._griffin_lim(S ** self.hps.power)

    def test(self, y, clip_norm=True):
        D = self._stft(y)
        src = np.abs(D)
        print('linear: ', np.min(src), np.max(src))
        mel_ = self._linear_to_mel(np.abs(D))
        print('mel_linear: ', np.min(mel_), np.max(mel_))
        mel_db = self._amp_to_db(mel_)
        print('mel_db: ', np.min(mel_db), np.max(mel_db))
        mel_db_ref = mel_db - self.hps.ref_level_db
        print('mel_db_ref: ', np.min(mel_db_ref), np.max(mel_db_ref))
        if clip_norm:
            mel_db_ref = self._normalize(mel_db_ref)
            print('mel_db_ref_norm: ', np.min(mel_db_ref), np.max(mel_db_ref))
            mel_db_ref_denorm = self._denormalize(mel_db_ref)
            print('mel_db_ref_denorm: ', np.min(mel_db_ref_denorm), np.max(mel_db_ref_denorm))
        else:
            mel_db_ref_denorm = mel_db_ref
        mel_db_de_ref = mel_db_ref_denorm + self.hps.ref_level_db
        print('mel_db_de_ref: ', np.min(mel_db_de_ref), np.max(mel_db_de_ref))
        mel_linear = self._db_to_amp(mel_db_de_ref)
        print('mel_linear_re: ', np.min(mel_linear), np.max(mel_linear))
        linear_sp = self._mel_to_linear(mel_linear)
        print('linear_re: ', np.min(linear_sp), np.max(linear_sp))

        print(np.mean(np.abs(src - linear_sp)))

    def melspectrogram(self, y, clip_norm=True):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hps.ref_level_db
        if clip_norm:
            S = self._normalize(S)
        return S

    def inv_mel_spectrogram(self, mel_spectrogram):
        S = self._mel_to_linear(self._db_to_amp(
            self._denormalize(mel_spectrogram) + self.hps.ref_level_db))
        return self._griffin_lim(S ** self.hps.power)

    def find_endpoint(self, wav, threshold_db=-40.0, min_silence_sec=0.8):
        window_length = int(self.hps.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x: x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.hps.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        if len(y.shape) == 1:  # [time_steps]
            return librosa.stft(y=y, n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                center=self.hps.center)
        elif len(y.shape) == 2:  # [batch_size, time_steps]
            if y.shape[0] == 1:  # batch_size=1
                return np.expand_dims(librosa.stft(y=y[0], n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   win_length=win_length,
                                                   center=self.hps.center),
                                      axis=0)
            else:  # batch_size > 1
                spec_list = list()
                for wav in y:
                    spec_list.append(librosa.stft(y=wav, n_fft=n_fft,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  center=self.hps.center))
                return np.concatenate(spec_list, axis=0)
        else:
            raise Exception('Wav dimension error in stft function!')

    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        if len(y.shape) == 2:  # spectrogram shape: [n_frame, n_fft]
            return librosa.istft(y, hop_length=hop_length,
                                 win_length=win_length,
                                 center=self.hps.center)
        elif len(y.shape) == 3:  # spectrogram shape: [batch_size, n_frame, n_fft]
            if y.shape[0] == 1:  # batch_size = 1
                return np.expand_dims(librosa.istft(y[0],
                                                    hop_length=hop_length,
                                                    win_length=win_length,
                                                    center=self.hps.center),
                                      axis=0)
            else:  # batch_size > 1
                wav_list = list()
                for spec in y:
                    wav_list.append(librosa.istft(spec,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  center=self.hps.center))
                    return np.concatenate(wav_list, axis=0)
        else:
            raise Exception('Spectrogram dimension error in istft function!')

    def _stft_parameters(self):
        n_fft = (self.hps.num_freq - 1) * 2
        # hop_length = int(self.hps.frame_shift_ms / 1000 * self.hps.sample_rate)
        # win_length = int(self.hps.frame_length_ms / 1000 * self.hps.sample_rate)
        hop_length = self.hps.frame_shift_sample
        win_length = self.hps.frame_length_sample
        return n_fft, hop_length, win_length

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spectrogram):
        _inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        linear_spectrogram = np.dot(_inv_mel_basis, mel_spectrogram)
        if len(linear_spectrogram.shape) == 3:
            # for 3-dimension mel, the shape of
            # inverse linear spectrogram will be [num_freq, batch_size, n_frame]
            linear_spectrogram = np.transpose(linear_spectrogram, [1, 0, 2])
        return np.maximum(1e-10, linear_spectrogram)

    def _build_mel_basis(self):
        n_fft = (self.hps.num_freq - 1) * 2
        return librosa.filters.mel(
            self.hps.sample_rate,
            n_fft=n_fft,
            n_mels=self.hps.num_mels,
            fmin=self.hps.min_mel_freq,
            fmax=self.hps.max_mel_freq)

    @staticmethod
    def _amp_to_db(x):
        return 20 * np.log10(np.maximum(1e-5, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        if self.hps.symmetric_specs:
            return np.clip(
                (2 * self.hps.max_abs_value) * (
                        (S - self.hps.min_level_db) / (-self.hps.min_level_db)
                ) - self.hps.max_abs_value,
                -self.hps.max_abs_value, self.hps.max_abs_value)
        else:
            return np.clip(self.hps.max_abs_value * (
                    (S - self.hps.min_level_db) / (-self.hps.min_level_db)),
                           0, self.hps.max_abs_value)

    def _denormalize(self, S):
        if self.hps.symmetric_specs:
            return ((np.clip(S, -self.hps.max_abs_value, self.hps.max_abs_value)
                     + self.hps.max_abs_value) * (-self.hps.min_level_db)
                    / (2 * self.hps.max_abs_value)
                    + self.hps.min_level_db)
        else:
            return ((np.clip(S, 0, self.hps.max_abs_value) * (-self.hps.min_level_db)
                     / self.hps.max_abs_value)
                    + self.hps.min_level_db)

    def preemphasize(self, x):
        if len(x.shape) == 1:  # [time_steps]
            return signal.lfilter([1, -self.hps.preemphasize], [1], x)
        elif len(x.shape) == 2:  # [batch_size, time_steps]
            if x.shape[0] == 1:
                return np.expand_dims(
                    signal.lfilter([1, -self.hps.preemphasize], [1], x[0]), axis=0)
            wav_list = list()
            for wav in x:
                wav_list.append(signal.lfilter([1, -self.hps.preemphasize], [1], wav))
            return np.concatenate(wav_list, axis=0)
        else:
            raise Exception('Wave dimension error in pre-emphasis')

    def inv_preemphasize(self, x):
        if self.hps.preemphasize is None:
            return x
        if len(x.shape) == 1:  # [time_steps]
            return signal.lfilter([1], [1, -self.hps.preemphasize], x)
        elif len(x.shape) == 2:  # [batch_size, time_steps]
            if x.shape[0] == 1:
                return np.expand_dims(
                    signal.lfilter([1], [1, -self.hps.preemphasize], x[0]), axis=0)
            wav_list = list()
            for wav in x:
                wav_list.append(signal.lfilter([1], [1, -self.hps.preemphasize], wav))
            return np.concatenate(wav_list, axis=0)
        else:
            raise Exception('Wave dimension error in inverse pre-emphasis')

    def mfcc(self, y):
        from scipy.fftpack import dct
        preemphasized = self.preemphasize(y)
        D = self._stft(preemphasized)
        S = librosa.power_to_db(self._linear_to_mel(np.abs(D) ** 2))
        mfcc = dct(x=S, axis=0, type=2, norm='ortho')[:self.hps.n_mfcc]
        deltas = librosa.feature.delta(mfcc)
        delta_deltas = librosa.feature.delta(mfcc, order=2)
        mfcc_feature = np.concatenate((mfcc, deltas, delta_deltas), axis=0)
        return mfcc_feature.T

    def hyper_parameters_estimation(self, wav_dir):
        from tqdm import tqdm
        wavs = []
        for root, dirs, files in os.walk(wav_dir):
            for f in files:
                if re.match(r'.+\.wav', f):
                    wavs.append(os.path.join(root, f))
        mel_db_min = 100.0
        mel_db_max = -100.0
        for f in tqdm(wavs):
            wav_arr = self.load_wav(f)
            pre_emphasized = self.preemphasize(wav_arr)
            D = self._stft(pre_emphasized)
            S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
            mel_db_max = np.max(S) if np.max(S) > mel_db_max else mel_db_max
            mel_db_min = np.min(S) if np.min(S) < mel_db_min else mel_db_min
        return mel_db_min, mel_db_max

    def _magnitude_spectrogram(self, audio, clip_norm):
        preemp_audio = self.preemphasize(audio)
        mel_spec = self.melspectrogram(preemp_audio, clip_norm)
        linear_spec = self.spectrogram(preemp_audio, clip_norm)
        return mel_spec.T, linear_spec.T

    def _energy_spectrogram(self, audio):
        preemp_audio = self.preemphasize(audio)
        linear_spec = np.abs(self._stft(preemp_audio)) ** 2
        mel_spec = self._linear_to_mel(linear_spec)
        return mel_spec.T, linear_spec.T

    def _extract_min_max(self, wav_path, mode, post_fn=lambda x: x):
        num_mels = self.hps.num_mels
        num_linears = self.hps.num_freq

        wavs = []
        for root, dirs, files in os.walk(wav_path):
            for f in files:
                if re.match(r'.+\.wav', f):
                    wavs.append(os.path.join(root, f))

        num_wavs = len(wavs)
        mel_mins_per_wave = np.zeros((num_wavs, num_mels))
        mel_maxs_per_wave = np.zeros((num_wavs, num_mels))
        linear_mins_per_wave = np.zeros((num_wavs, num_linears))
        linear_maxs_per_wave = np.zeros((num_wavs, num_linears))

        for i, wav in enumerate(post_fn(wavs)):
            audio, sr = soundfile.read(wav)
            if mode == 'magnitude':
                mel, linear = self._magnitude_spectrogram(audio, clip_norm=False)
            elif mode == 'energy':
                mel, linear = self._energy_spectrogram(audio)
            else:
                raise Exception('Only magnitude or energy is supported!')

            mel_mins_per_wave[i,] = np.amin(mel, axis=0)
            mel_maxs_per_wave[i,] = np.amax(mel, axis=0)
            linear_mins_per_wave[i,] = np.amin(linear, axis=0)
            linear_maxs_per_wave[i,] = np.amax(linear, axis=0)

        mel_mins = np.reshape(np.amin(mel_mins_per_wave, axis=0), (1, num_mels))
        mel_maxs = np.reshape(np.amax(mel_maxs_per_wave, axis=0), (1, num_mels))
        linear_mins = np.reshape(np.amin(linear_mins_per_wave, axis=0), (1, num_mels))
        linear_maxs = np.reshape(np.amax(linear_mins_per_wave, axis=0), (1, num_mels))
        min_max = {
            'mel_min': mel_mins,
            'mel_max': mel_maxs,
            'linear_mins': linear_mins,
            'linear_max': linear_maxs
        }
        return min_max

    @staticmethod
    def _normalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
        spec_dim = len(spec.T)
        num_frame = len(spec)

        max_min = maxs - mins
        max_min = np.reshape(max_min, (1, spec_dim))
        max_min[max_min <= 0.0] = 1.0

        target_max_min = np.zeros((1, spec_dim))
        target_max_min.fill(max_value - min_value)
        target_max_min[max_min <= 0.0] = 1.0

        spec_min = np.tile(mins, (num_frame, 1))
        target_min = np.tile(min_value, (num_frame, spec_dim))
        spec_range = np.tile(max_min, (num_frame, 1))
        norm_spec = np.tile(target_max_min, (num_frame, 1)) / spec_range
        norm_spec = norm_spec * (spec - spec_min) + target_min
        return norm_spec

    @staticmethod
    def _denormalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
        spec_dim = len(spec.T)
        num_frame = len(spec)

        max_min = maxs - mins
        max_min = np.reshape(max_min, (1, spec_dim))
        max_min[max_min <= 0.0] = 1.0

        target_max_min = np.zeros((1, spec_dim))
        target_max_min.fill(max_value - min_value)
        target_max_min[max_min <= 0.0] = 1.0

        spec_min = np.tile(mins, (num_frame, 1))
        target_min = np.tile(min_value, (num_frame, spec_dim))
        spec_range = np.tile(max_min, (num_frame, 1))
        denorm_spec = spec_range / np.tile(target_max_min, (num_frame, 1))
        denorm_spec = denorm_spec * (spec - target_min) + spec_min
        return denorm_spec

    @staticmethod
    def rescale(mel):
        x = np.linspace(1, mel.shape[0], mel.shape[0])
        xn = np.linspace(1, mel.shape[0], int(mel.shape[0] * 1.25))
        f = interpolate.interp1d(x, mel, kind='cubic', axis=0)
        rescaled_mel = f(xn)
        return rescaled_mel
