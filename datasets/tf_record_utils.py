import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm


class TFRecordWriter:
    def __init__(self, train_split=None, data_dir=None, save_dir=None):
        self.train_split = train_split
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.train_ids_file = os.path.join(self.data_dir, 'train.txt') if data_dir is not None else None
        self.dev_ids_file = os.path.join(self.data_dir, 'dev.txt') if data_dir is not None else None
        self.test_ids_file = os.path.join(self.data_dir, 'test.txt') if data_dir is not None else None

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize_example(fid, text, mel, text_len, mel_len):
        """
        :param fid: string
        :param text: character id list
        :param mel: np array, [mel_len, num_mels]
        :param text_len: int32
        :param mel_len: int32
        :return: byte string
        """
        feature = {
            'fid': TFRecordWriter._bytes_feature(fid.encode('utf-8')),
            'text': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(text)),
            'mel': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(mel)),
            'text_len': TFRecordWriter._int64_feature(text_len),
            'mel_len': TFRecordWriter._int64_feature(mel_len),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def _parse_fids(self, mode='train'):
        fids_f = {'train': self.train_ids_file,
                  'dev': self.dev_ids_file,
                  'test': self.test_ids_file}[mode]
        fids = []
        with open(fids_f, 'r', encoding='utf-8') as f:
            for line in f:
                fids.append(line.strip())
        return fids

    def _get_features(self, fid):
        text = np.load(os.path.join(self.data_dir, 'texts', '{}.npy'.format(fid)))
        mel = np.load(os.path.join(self.data_dir, 'mels', '{}.npy'.format(fid)))
        text_len = len(text)
        mel_len = mel.shape[0]
        return text, mel, text_len, mel_len

    def write(self, mode='train'):
        fids = self._parse_fids(mode)
        if mode == 'train':
            splited_fids = [fids[i::self.train_split] for i in range(self.train_split)]
        else:
            splited_fids = [fids]
        for i, ids in enumerate(splited_fids):
            tfrecord_path = os.path.join(self.save_dir, '{}-{}.tfrecords'.format(mode, i))
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for fid in tqdm(ids):
                    text, mel, text_len, mel_len = self._get_features(fid)
                    serialized_example = self.serialize_example(fid, text, mel, text_len, mel_len)
                    writer.write(serialized_example)
        return

    def write_all(self):
        self.write('train')
        self.write('dev')
        self.write('test')
        return

    def pre_pad(self, inputs):
        """
        :param inputs: [nframe, dim]
        :return:
        """
        nframe = tf.shape(inputs)[0]
        if self.pad_factor == 0 or self.pad_factor == 1:
            return inputs
        if nframe % self.pad_factor != 0:
            paddings = tf.zeros_like(inputs)[: self.pad_factor - nframe % self.pad_factor, :]
            padded = tf.concat([inputs, paddings], axis=0)
        else:
            padded = inputs
        return padded

    def parse_example(self, serialized_example):
        feature_description = {
            'fid': tf.io.FixedLenFeature((), tf.string),
            'text': tf.io.FixedLenFeature((), tf.string),
            'mel': tf.io.FixedLenFeature((), tf.string),
            'text_len': tf.io.FixedLenFeature((), tf.int64),
            'mel_len': tf.io.FixedLenFeature((), tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)

        fid = example['fid']
        text = tf.io.parse_tensor(example['text'], out_type=tf.int64)
        mel = tf.io.parse_tensor(example['mel'], out_type=tf.float64)
        mel = self.pre_pad(mel)
        text_len = example['text_len']
        mel_len = example['mel_len']
        return fid, tf.cast(text, tf.int32), tf.cast(mel, tf.float32), tf.cast(text_len, tf.int32), tf.cast(mel_len, tf.int32)

    def create_dataset(self, buffer_size, num_parallel_reads, pad_factor,
                       batch_size, num_mels, shuffle_buffer, shuffle,
                       tfrecord_files, seed=1):
        tfrecord_dataset = tf.data.TFRecordDataset(
            tfrecord_files, buffer_size=buffer_size,
            num_parallel_reads=num_parallel_reads)
        self.pad_factor = pad_factor
        ljdataset = tfrecord_dataset.map(
            self.parse_example,
            num_parallel_calls=num_parallel_reads)
        ljdataset = ljdataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([], [None], [None, num_mels], [], []))
        ljdataset = (ljdataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
                     if shuffle else ljdataset)
        ljdataset = ljdataset.prefetch(tf.data.experimental.AUTOTUNE)
        return ljdataset

    def get_tfrecords_list(self, mode='train'):
        assert self.save_dir is not None
        assert mode in ['train', 'dev', 'test']
        return [os.path.join(self.save_dir, f)
                for f in os.listdir(self.save_dir) if f.startswith(mode)]


def write_lj_tfrecord():
    ljrecord = TFRecordWriter(train_split=8,
                              data_dir='./',
                              save_dir='./tfrecords')
    ljrecord.write_all()


def tf_record_test():
    ljrecords = TFRecordWriter(save_dir='./tfrecords')
    # ljrecord.write(mode='dev')
    records = ['./tfrecords/test-0.tfrecords']

    class Hparams:
        class Train:
            train_batch_size = 64
            shuffle_buffer = 128
            shuffle = False

        class Audio:
            num_mels = 80

        class Dataset:
            buffer_size = 65536
            num_parallel_reads = 32
            pad_factor = 10  # factor ** (num_blk - 1)

    hparams = Hparams()
    ljdataset = ljrecords.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        pad_factor=hparams.Dataset.pad_factor,
        batch_size=hparams.Train.train_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=records)
    for epoch in range(2):
        for i, data in enumerate(ljdataset):
            print('epoch {}, step: {}'.format(epoch, i))
            fid, text, mel, text_len, mel_len = data
            print(fid)


if __name__ == '__main__':
    tf_record_test()
