import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from audio import Audio
from pypinyin import pinyin, Style
from texts import english_cleaners


class TextMelData:
    def __init__(self, data_root, save_dir, hps):
        self.data_root = data_root
        self.save_dir = save_dir
        self.hps = hps
        self.text_dict_f = os.path.join(self.save_dir, 'texts.pkl')
        self.mel_dir = os.path.join(self.save_dir, 'mels')
        self.text_dir = os.path.join(self.save_dir, 'texts')
        self.train_list_f = os.path.join(self.save_dir, 'train.txt')
        self.dev_list_f = os.path.join(self.save_dir, 'dev.txt')
        self.test_list_f = os.path.join(self.save_dir, 'test.txt')
        self.dev_set_rate = hps.Dataset.dev_set_rate
        self.test_set_rate = hps.Dataset.test_set_rate
        self.num_mels = hps.Audio.num_mels
        self.audio_processor = Audio(hps.Audio)
        self.batch_size = hps.Train.train_batch_size
        self.text_dict = None
        self.train_set_size = None
        self.dev_set_size = None
        self.test_set_size = None
        self.train_generator = None
        self.dev_generator = None

    def feature_extraction(self):
        if self.feats_extract_finish():
            print('Features already exists!')
            with open(self.text_dict_f, 'rb') as f:
                self.text_dict = pickle.load(f)
                self.train_set_size = self._count_file_lines(self.train_list_f)
                self.dev_set_size = self._count_file_lines(self.dev_list_f)
                self.test_set_size = self._count_file_lines(self.test_list_f)
        else:
            self._validate_dir()
            print('Process text file...')
            self.text_dict = self.text_process()
            print('Split the data set into train, dev and test set...')
            self.train_set_size, self.dev_set_size, self.test_set_size = self.dataset_split()
            print('Extracting Mel-Spectrograms...')
            self.extract_mels()
        return

    def get_train_dev_set(self):
        # create tf dataset
        self.train_generator = self.get_generator('train')
        self.dev_generator = self.get_generator('dev')
        return self.train_generator, self.dev_generator

    def _validate_dir(self):
        assert os.path.isdir(self.data_root)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(self.mel_dir):
            os.makedirs(self.mel_dir)
        if not os.path.isdir(self.text_dir):
            os.makedirs(self.text_dir)
        return

    @staticmethod
    def _count_file_lines(f):
        c = 0
        with open(f, 'r') as f:
            for _ in f:
                c += 1
        return c

    def feats_extract_finish(self):
        if ((not os.path.exists(self.text_dict_f))
                or (not os.path.isfile(self.train_list_f))
                or (not os.path.isfile(self.dev_list_f))
                or (not os.path.isfile(self.test_list_f))
                or (not os.path.isdir(self.mel_dir))
                or (not os.path.isdir(self.text_dir))):
            print('Some features or files not existing, extracting from scratch ... ')
            return False
        else:
            for f in [self.train_list_f,
                      self.dev_list_f,
                      self.test_list_f]:
                with open(f, 'r') as rf:
                    for line in rf:
                        utt_id = line.strip()
                        if not os.path.isfile(
                                os.path.join(
                                    self.mel_dir, '{}.npy'.format(utt_id))):
                            print('{} not exists!'.format(
                                os.path.join(self.mel_dir, '{}.npy'.format(utt_id))))
                            return False
            return True

    def dataset_split(self):
        with open(self.text_dict_f, 'rb') as f:
            text_dict = pickle.load(f)
        dev_set = []
        test_set = []
        # sort by utterance lengths
        utt_ids = [k for k, t in
                   sorted(text_dict.items(), key=lambda x: len(x[1]))]
        data_size = len(utt_ids)
        dev_size = int(self.dev_set_rate * data_size)
        test_size = int(self.test_set_rate * data_size)
        # sample dev set from the whole data
        dev_rate = data_size // dev_size
        for i in range(0, data_size, dev_rate):
            sample = np.random.choice(utt_ids[i: i + dev_rate], 1)[0]
            dev_set.append(sample)
        for item in dev_set:
            utt_ids.remove(item)
        # sample test set from the remaining data
        data_size = len(utt_ids)
        test_rate = data_size // test_size
        for i in range(0, data_size, test_rate):
            sample = np.random.choice(utt_ids[i: i + test_rate], 1)[0]
            test_set.append(sample)
        for item in test_set:
            utt_ids.remove(item)
        train_set = utt_ids
        # write splitted dataset to readable files
        with open(self.train_list_f, 'w') as f:
            for idx in train_set:
                f.write("{}\n".format(idx))
        with open(self.dev_list_f, 'w') as f:
            for idx in dev_set:
                f.write("{}\n".format(idx))
        with open(self.test_list_f, 'w') as f:
            for idx in test_set:
                f.write("{}\n".format(idx))
        return len(train_set), len(dev_set), len(test_set)

    def get_wav_files(self, ext='.wav'):
        wav_files = []
        for root, dirs, files in os.walk(self.data_root, followlinks=True):
            for basename in files:
                if basename.endswith(ext):
                    filename = os.path.join(root, basename)
                    wav_files.append(filename)
        return wav_files

    def extract_mels(self):
        wav_list = self.get_wav_files()
        for wav_f in tqdm(wav_list):
            wav_arr = self.audio_processor.load_wav(wav_f)
            wav_arr = self.audio_processor.preemphasize(wav_arr)
            mels = self.audio_processor.melspectrogram(wav_arr)
            fid = wav_f.split('/')[-1].split('.')[0]
            save_name = os.path.join(self.mel_dir, fid + '.npy')
            np.save(save_name, mels.T)
        return

    def text_process(self):
        """
        Should implement this function according to the way how transcripts
        are organized in the dataset.

        This function does the text analysis to the transcripts, i.e., normalization,
         g2p, etc, and transform the texts into numeric sequences.

        Then saves text npy file per utterance with file name
        os.path.join(self.text_dir, '{}.npy'.format(utt_id))

        :return: the text dict where the key represents the utterance id,
        the value represents the
        """
        # np.save(os.path.join(self.text_dir, '{}.npy'.format(lst[0])), seq)
        # with open(self.text_dict_f, 'wb') as f:
        #     pickle.dump(text_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        # return text_dict
        pass

    def get_batch(self, ids_file, rank=None, size=None):
        text_pad = 0
        mel_pad = np.zeros([1, self.num_mels], dtype=np.float32)
        with open(ids_file, 'r') as f:
            utt_ids = []
            for line in f:
                utt_id = line.strip()
                utt_ids.append(utt_id)
        # possible split
        if rank is not None and size is not None:
            if rank > size:
                raise ValueError(
                    'rank should be smaller than size! rank: {}, size: {}'.format(rank, size))
            utt_ids = utt_ids[rank::size]
        for i in range(0, len(utt_ids), self.batch_size):
            text_batch = []
            mel_batch = []
            text_len_batch = []
            mel_len_batch = []
            utt_id_batch = []
            for j in range(self.batch_size):
                if i + j >= len(utt_ids):
                    break
                utt_id = utt_ids[i + j]
                text_seq = self.text_dict[utt_id]
                text_batch.append(text_seq)
                mel = np.load(os.path.join(self.mel_dir, '{}.npy'.format(utt_id)))
                mel_batch.append(mel)
                text_len = len(text_seq)
                text_len_batch.append(text_len)
                mel_len = mel.shape[0]
                mel_len_batch.append(mel_len)
                utt_id_batch.append(utt_id)
            # padded batch
            text_max_len = max(text_len_batch)
            mel_max_len = max(mel_len_batch)
            for j, text in enumerate(text_batch):
                if len(text) < text_max_len:
                    text_batch[j] += [text_pad] * (text_max_len - len(text))
            text_batch = np.stack(text_batch, axis=0)
            for j, mel in enumerate(mel_batch):
                if mel.shape[0] < mel_max_len:
                    num_pad = mel_max_len - mel.shape[0]
                    padding = np.tile(mel_pad, (num_pad, 1))
                    mel_batch[j] = np.concatenate((mel, padding), axis=0)
            mel_batch = np.stack(mel_batch, axis=0)
            yield text_batch, mel_batch, np.array(text_len_batch, dtype=np.int32), \
                  np.array(mel_len_batch, dtype=np.int32), utt_id_batch

    def get_generator(self, mode='train', rank=None, size=None):
        assert mode in ['train', 'dev', 'test']
        return {'train': self.get_batch(self.train_list_f, rank, size),
                'dev': self.get_batch(self.dev_list_f, rank, size),
                'test': self.get_batch(self.test_list_f, rank, size)}[mode]


class LJSpeech(TextMelData):
    def __init__(self, data_root, save_dir, hps):
        super(LJSpeech, self).__init__(data_root=data_root, save_dir=save_dir, hps=hps)

    def text_process(self):
        text_f = os.path.join(self.data_root, 'metadata.csv')
        text_dict = {}
        with open(text_f, 'r', encoding='utf-8') as rf:
            for line in rf:
                s = line.strip()
                lst = s.split('|')
                seq = self.text_to_array(lst[2])
                text_dict[lst[0]] = seq
                np.save(os.path.join(self.text_dir, '{}.npy'.format(lst[0])), seq)
        with open(self.text_dict_f, 'wb') as f:
            pickle.dump(text_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return text_dict

    def text_to_array(self, text):
        # text normalization
        text = english_cleaners(text)
        _bos = self.hps.Texts.bos
        _eos = self.hps.Texts.eos
        text = _bos + text + _eos
        symbols = list(self.hps.Texts.characters)
        _symbol_to_id = {s: i for i, s in enumerate(symbols)}
        text_arr = [_symbol_to_id[s] for s in text]
        return text_arr


class DataBaker(TextMelData):
    def __init__(self, data_root, save_dir, hps):
        super(DataBaker, self).__init__(data_root=data_root, save_dir=save_dir, hps=hps)

    @staticmethod
    def _is_erhua(pinyin_seq):
        """
        Decide whether pinyin (without tone number) is retroflex (Erhua)
        """
        if len(pinyin_seq) <= 1 or pinyin_seq == 'er':
            return False
        elif pinyin_seq[-1] == 'r':
            return True
        else:
            return False

    def _parse_cn_prosody_label(self, text, pinyin_seq, use_prosody=False):
        """
        Parse label from text and pronunciation lines with prosodic structure labelings

        Input text:    100001 妈妈#1当时#1表示#3，儿子#1开心得#2像花儿#1一样#4。
        Input pinyin:  ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4
        Return sen_id: 100001
        Return pinyin: ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.
        Args:
            - text: Chinese characters with prosodic structure labeling, begin with sentence id for wav and interval file
            - pinyin: Pinyin pronunciations, with tone 1-5
            - use_prosody: Whether the prosodic structure labeling information will be used
        Returns:
            - (sen_id, pinyin&tag): latter contains pinyin string with optional prosodic structure tags
        """

        text = text.strip()
        pinyin_seq = pinyin_seq.strip()
        if len(text) == 0:
            return None

        # remove punctuations
        text = re.sub('[“”、，。：；？！—…#（）]', '', text)

        # split into sub-terms
        sen_id, texts = text.split()
        phones = pinyin_seq.split()

        # prosody boundary tag (SYL: 音节, PWD: 韵律词, PPH: 韵律短语, IPH: 语调短语, SEN: 语句)
        SYL = '-'
        PWD = ' '
        PPH = ' / ' if use_prosody is True else ' '
        IPH = ', '
        SEN = '.'
        # parse details
        py_seq = ''
        i = 0  # texts index
        j = 0  # phones index
        b = 1  # left is boundary
        while i < len(texts):
            if texts[i].isdigit():
                if texts[i] == '1':
                    py_seq += PWD  # Prosodic Word, 韵律词边界
                if texts[i] == '2':
                    py_seq += PPH  # Prosodic Phrase, 韵律短语边界
                if texts[i] == '3':
                    py_seq += IPH  # Intonation Phrase, 语调短语边界
                if texts[i] == '4':
                    py_seq += SEN  # Sentence, 语句结束
                b = 1
                i += 1
            elif texts[i] != '儿' or j == 0 or not self._is_erhua(phones[j - 1][:-1]):  # Chinese segment
                if b == 0:
                    py_seq += SYL  # Syllable, 音节边界（韵律词内部）
                py_seq += phones[j]
                b = 0
                i += 1
                j += 1
            else:  # 儿化音
                i += 1
        return py_seq

    def text_process(self):
        symbols = list(self.hps.Texts.characters)
        _bos = self.hps.Texts.bos
        _eos = self.hps.Texts.eos
        _symbol_to_id = {s: i for i, s in enumerate(symbols)}
        text_file = os.path.join(self.data_root, '000001-010000.txt')
        text_dict = {}
        with open(text_file, mode='r', encoding='utf-8') as f:
            for line in f:
                if line[0].isdigit():
                    fid = line[:6]
                    text = line
                else:
                    pinyin_seq = self._parse_cn_prosody_label(text, line)
                    sent = _bos + pinyin_seq.lower() + _eos
                    seq = [_symbol_to_id[s] for s in sent]
                    text_dict[fid] = seq
                    np.save(os.path.join(self.text_dir, '{}.npy'.format(fid)), seq)
        with open(self.text_dict_f, 'wb') as f:
            pickle.dump(text_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return text_dict

    def text_to_array(self, text):
        symbols = list(self.hps.Texts.characters)
        _bos = self.hps.Texts.bos
        _eos = self.hps.Texts.eos
        _symbol_to_id = {s: i for i, s in enumerate(symbols)}
        py = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors='ignore')
        sent = ''
        for i, p in enumerate(py):
            sent += p[0].lower()
            if i != len(py) - 1:
                sent += ' '
        sent = _bos + sent + _eos
        seq = [_symbol_to_id[s] for s in sent]
        return seq
