import os
import random
import argparse
import numpy as np
from datasets import LJSpeech, DataBaker, TFRecordWriter
from configs import LJHPS, DataBakerHPS


dataset_hps = {'ljspeech': LJHPS, 'databaker': DataBakerHPS}
dataset_extractor = {'ljspeech': LJSpeech, 'databaker': DataBaker}


def main():
    parser = argparse.ArgumentParser('Data preprocessing')
    parser.add_argument('--dataset', type=str, choices=['ljspeech', 'databaker'],
                        help='dataset name, currently support ljspeech and databaker')
    parser.add_argument('--data_dir', type=str, help='data root directory')
    parser.add_argument('--save_dir', type=str, help='feature save directory')
    parser.add_argument('--record_split', type=int,
                        help='number of train tf-record to be split', default=8)
    args = parser.parse_args()
    hps = dataset_hps[args.dataset]
    random.seed(hps.Train.random_seed)
    np.random.seed(hps.Train.random_seed)
    feats_extractor = dataset_extractor[args.dataset](data_root=args.data_dir, save_dir=args.save_dir, hps=hps)
    feats_extractor.feature_extraction()
    tfrecord_save_dir = os.path.join(args.save_dir, 'tfrecords')
    if not os.path.exists(tfrecord_save_dir):
        os.makedirs(tfrecord_save_dir)
    tfrecord_writer = TFRecordWriter(train_split=args.record_split, data_dir=args.save_dir, save_dir=tfrecord_save_dir)
    tfrecord_writer.write_all()

    # test
    # 1. Mel test
    print('Basic dataset information ...')
    print('Training set size: {}'.format(feats_extractor.train_set_size))
    print('Validation set size: {}'.format(feats_extractor.dev_set_size))
    print('Test set size: {}'.format(feats_extractor.test_set_size))
    test_generator = feats_extractor.get_generator('test')
    text_batch, mel_batch, text_len, mel_len, utt_ids = next(test_generator)
    print('Text shape: {}'.format(text_batch.shape))
    print('Mel shape: {}'.format(mel_batch.shape))
    print('Text lengths: {}'.format(text_len))
    print('Mel lengths: {}'.format(mel_len))
    print('Utterance IDs: {}'.format(utt_ids))
    print('--------------------------------------------------------')
    print('TFRecord test...')
    tf_dataset = tfrecord_writer.create_dataset(
        buffer_size=hps.Dataset.buffer_size,
        num_parallel_reads=hps.Dataset.num_parallel_reads,
        pad_factor=hps.Dataset.pad_factor,
        batch_size=hps.Train.test_batch_size,
        num_mels=hps.Audio.num_mels,
        shuffle_buffer=hps.Train.shuffle_buffer,
        shuffle=hps.Train.shuffle,
        tfrecord_files=tfrecord_writer.get_tfrecords_list('test'))
    for epoch in range(2):
        for i, data in enumerate(tf_dataset):
            print('epoch {}, step: {}'.format(epoch, i))
            fid, text, mel, text_len, mel_len = data
            print(fid.numpy(), text.shape, mel.shape, text_len, mel_len)


def tfrecord_test():
    hps = LJHPS()
    tfrecord_writer = TFRecordWriter(save_dir='./')
    tf_records = tfrecord_writer.get_tfrecords_list('test')
    tf_dataset = tfrecord_writer.create_dataset(
        buffer_size=hps.Dataset.buffer_size,
        num_parallel_reads=hps.Dataset.num_parallel_reads,
        pad_factor=hps.Dataset.pad_factor,
        batch_size=hps.Train.test_batch_size,
        num_mels=hps.Audio.num_mels,
        shuffle_buffer=hps.Train.shuffle_buffer,
        shuffle=hps.Train.shuffle,
        tfrecord_files=tf_records)
    for epoch in range(2):
        for i, data in enumerate(tf_dataset):
            print('epoch {}, step: {}'.format(epoch, i))
            fid, text, mel, text_len, mel_len = data
            print(fid.numpy(), text.numpy, mel.shape, text_len, mel_len)


if __name__ == '__main__':
    main()
