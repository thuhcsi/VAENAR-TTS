import os
import sys
import random
import argparse
import datetime
import numpy as np
import tensorflow as tf

from time import time

from models import VAENAR
from audio import TestUtils
from datasets import TFRecordWriter
from configs import LJHPS, DataBakerHPS, Logger


def main():
    parser = argparse.ArgumentParser('Training parameters parser')
    parser.add_argument('--dataset', type=str, choices=['ljspeech', 'databaker'],
                        help='dataset name, currently support ljspeech and databaker')
    parser.add_argument('--data_dir', type=str,
                        help='dataset tfrecord directory')
    parser.add_argument('--model_dir', type=str,
                        help='directory to save model ckpt')
    parser.add_argument('--log_dir', type=str,
                        help='directory to save log')
    parser.add_argument('--test_dir', type=str,
                        help='directory to save test results',
                        default=None)
    args = parser.parse_args()

    hparams = {'ljspeech': LJHPS, 'databaker': DataBakerHPS}[args.dataset]
    # set random seed
    random.seed(hparams.Train.random_seed)
    np.random.seed(hparams.Train.random_seed)
    tf.random.set_seed(hparams.Train.random_seed)

    # set up test utils
    tester = TestUtils(hparams, args.test_dir)

    # validate log directories
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    # set up logger
    sys.stdout = Logger(log_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join(log_dir, current_time, 'train')
    os.makedirs(train_dir)
    dev_dir = os.path.join(log_dir, current_time, 'dev')
    os.makedirs(dev_dir)

    # hyperparameters
    ljrecords = TFRecordWriter(save_dir=args.data_dir)
    train_set = ljrecords.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        pad_factor=hparams.Dataset.pad_factor,
        batch_size=hparams.Train.train_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=ljrecords.get_tfrecords_list('train'),
        seed=hparams.Train.random_seed)
    dev_set = ljrecords.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        pad_factor=hparams.Dataset.pad_factor,
        batch_size=hparams.Train.train_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=ljrecords.get_tfrecords_list('dev'),
        seed=hparams.Train.random_seed)
    test_set = ljrecords.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        pad_factor=hparams.Dataset.pad_factor,
        batch_size=hparams.Train.test_batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=ljrecords.get_tfrecords_list('test'),
        seed=hparams.Train.random_seed)

    # 2. setup model
    model = VAENAR(hparams)
    learning_rate = hparams.Train.learning_rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    # 3. define training step
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])
    def train_step(texts, mels, t_lengths, m_lengths, kl_weight, reduction_factor):
        print('tracing back at train_step')
        with tf.GradientTape() as tape:
            # predictions, per_example_l2, per_example_kl, per_example_len_l2, dec_alignments
            predictions, mel_l2, kl_divergence, length_l2, dec_alignments = model(
                inputs=texts, mel_targets=mels, mel_lengths=m_lengths,
                text_lengths=t_lengths, reduction_factor=reduction_factor,
                training=True, reduce_loss=True)
            loss = mel_l2 + kl_weight * tf.math.maximum(kl_divergence, 0.) + hparams.Train.length_weight * length_l2
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, mel_l2, kl_divergence, length_l2

    # 4. define validate step
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])
    def dev_step(texts, mels, t_lengths, m_lengths, kl_weight, reduction_factor):
        print('tracing back at dev step')
        predictions, mel_l2, kl_divergence, length_l2, dec_alignments = model(
            inputs=texts, mel_targets=mels, mel_lengths=m_lengths,
            text_lengths=t_lengths, reduction_factor=reduction_factor,
            training=False, reduce_loss=True)
        loss = mel_l2 + kl_weight * kl_divergence + hparams.Train.length_weight * length_l2
        return loss, mel_l2, kl_divergence, length_l2

    # 5. define test step
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])
    def test_step(texts, t_lengths, m_lengths, reduction_factor):
        print('tracing back at test step')
        predictions, dec_alignments = model.inference(inputs=texts,
                                                      mel_lengths=m_lengths,
                                                      text_lengths=t_lengths,
                                                      reduction_factor=reduction_factor)
        return predictions, dec_alignments

    # 6. define initiate step
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def init_step(texts, t_lengths, m_lengths):
        print('tracing back at init step')
        outputs = model.init(text_inputs=texts, mel_lengths=m_lengths, text_lengths=t_lengths)
        return outputs

    # @tf.function
    def train_one_epoch(dataset, kl_weight, reduction_factor):
        # print('tracing back at train_one_epoch')
        step = 0
        total = 0.0
        mel_l2 = 0.0
        kl = 0.0
        len_l2 = 0.0
        for _, train_texts, train_mels, train_t_lengths, train_m_lengths in dataset:
            step_start = time()
            _total, _mel_l2, _kl, _len_l2 = train_step(
                train_texts, train_mels, train_t_lengths, train_m_lengths,
                tf.constant(kl_weight), tf.constant(reduction_factor))
            step_end = time()
            # tf.print('Step', step, ': total', _total, 'l2', _l2, 'kl', _kl, 'time',
            #          step_end - step_start, end='\r')
            print('Step {}: total {:.6f}, mel-l2 {:.6f}, kl {:.3f}, len-l2 {:.3f}, time {:.3f}'.format(
                step, _total.numpy(), _mel_l2.numpy(), _kl.numpy(), _len_l2.numpy(), step_end - step_start))
            step += 1
            total += _total.numpy()
            mel_l2 += _mel_l2.numpy()
            kl += _kl.numpy()
            len_l2 += _len_l2.numpy()
        return total / step, mel_l2 / step, kl / step, len_l2 / step

    # @tf.function
    def dev_one_epoch(dataset, kl_weight, reduction_factor):
        # print('tracing back at dev_one_epoch')
        step = 0
        total = 0.0
        mel_l2 = 0.0
        kl = 0.0
        len_l2 = 0.0
        for _, dev_texts, dev_mels, dev_t_lengths, dev_m_lengths in dataset:
            _total, _mel_l2, _kl, _len_l2 = dev_step(
                dev_texts, dev_mels, dev_t_lengths, dev_m_lengths,
                tf.constant(kl_weight), tf.constant(reduction_factor))
            step += 1
            total += _total.numpy()
            mel_l2 += _mel_l2.numpy()
            kl += _kl.numpy()
            len_l2 += _len_l2.numpy()
        return total / step, mel_l2 / step, kl / step, len_l2 / step

    # 8. setup summary writer
    train_summary_writer = tf.summary.create_file_writer(train_dir)
    dev_summary_writer = tf.summary.create_file_writer(dev_dir)

    # kl weight computation
    kl_weight_init = hparams.Train.kl_weight_init
    kl_weight_end = hparams.Train.kl_weight_end
    kl_weight_inc_epochs = hparams.Train.kl_weight_increase_epoch
    kl_weight_step = (kl_weight_end - kl_weight_init) / kl_weight_inc_epochs

    # reduction factor computation
    def _get_reduction_factor(ep):
        intervals = hparams.Train.reduce_interval
        rfs = hparams.Train.reduction_factors
        i = 0
        while i < len(intervals) and intervals[i] <= ep:
            i += 1
        i = i - 1 if i > 0 else 0
        return rfs[i]

    # 9. setup checkpoint: all workers will need checkpoint manager to load checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64, trainable=False),
                                     optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=20, keep_checkpoint_every_n_hours=4)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        step = checkpoint.step.numpy()
    else:
        print("Initializing from scratch.")
        step = 0
        # initiate some parameters
        for fids, texts, mels, t_lengths, m_lengths in train_set.take(1):
            _ = init_step(texts, t_lengths, m_lengths)
            # save initial model
            save_path = manager.save()
            print("Initial checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            _total, _mel_l2, _kl, _len_l2 = train_step(
                texts, mels, t_lengths, m_lengths, tf.constant(kl_weight_init),
                tf.constant(hparams.Common.max_reduction_factor))
            print('Initial step: total {:.6f}, mel-l2 {:.6f}, kl {:.3f}, len-l2 {:.3f}'.format(
                _total.numpy(), _mel_l2.numpy(), _kl.numpy(), _len_l2.numpy()))

    # 8. start training
    for epoch in range(step + 1, hparams.Train.epochs):
        train_kl_weight = kl_weight_init + kl_weight_step * epoch if epoch <= kl_weight_inc_epochs else kl_weight_end
        reduction_factor = _get_reduction_factor(epoch)
        print('Training Epoch {}, kl weight is {}, reduction factor is {}...'.format(
            epoch, train_kl_weight, reduction_factor))
        epoch_start = time()
        train_total, train_mel_l2, train_kl, train_len_l2 = train_one_epoch(
            train_set, train_kl_weight, reduction_factor)
        epoch_dur = time() - epoch_start
        print('\nTraining Epoch {} finished in {:.3f} Secs'.format(epoch, epoch_dur))
        # save summary and evaluate
        with train_summary_writer.as_default():
            tf.summary.scalar('total-loss', train_total, step=epoch)
            tf.summary.scalar('recon-loss', train_mel_l2, step=epoch)
            tf.summary.scalar('kl-loss', train_kl, step=epoch)
            tf.summary.scalar('length-loss', train_len_l2, step=epoch)

        # validation
        print('Validation ...')
        dev_start = time()
        dev_total, dev_mel_l2, dev_kl, dev_len_l2 = dev_one_epoch(
            dev_set, train_kl_weight, reduction_factor)
        print('Validation finished in {:.3f} Secs'.format(time() - dev_start))
        with dev_summary_writer.as_default():
            tf.summary.scalar('total-loss', dev_total, step=epoch)
            tf.summary.scalar('recon-loss', dev_mel_l2, step=epoch)
            tf.summary.scalar('kl-loss', dev_kl, step=epoch)
            tf.summary.scalar('length-loss', dev_len_l2, step=epoch)

        print('Epoch {}:  train-total {}, train-mel-l2 {}, train-kl {},'
              'train-len-l2 {}, dev-total {}, dev-l2 {}, dev-kl {}, dev-len-l2 {}'.format(
            epoch, train_total, train_mel_l2, train_kl, train_len_l2,
            dev_total, dev_mel_l2, dev_kl, dev_len_l2))

        # save checkpoint
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.step), save_path))
        checkpoint.step.assign_add(1)

        # test
        if epoch % hparams.Train.test_interval == 0:
            print('Testing ...')
            for test_ids, test_texts, _, test_t_lengths, test_m_lengths in test_set.take(1):
                test_predicted_mel, test_dec_ali = test_step(
                    test_texts, test_t_lengths, test_m_lengths, tf.constant(reduction_factor))
                try:
                    tester.synthesize_and_save_wavs(epoch, test_predicted_mel.numpy(),
                                                    test_m_lengths.numpy(), test_ids.numpy(), 'test')
                except:
                    print('Something wrong with the generated waveform!')
                tester.draw_melspectrograms(epoch, test_predicted_mel.numpy(),
                                            test_m_lengths.numpy(), test_ids.numpy(), 'test')
                for k in test_dec_ali.keys():
                    tester.multi_draw_attention_alignments(
                        test_dec_ali[k].numpy(), test_texts.numpy(), test_t_lengths.numpy(),
                        test_m_lengths.numpy(), epoch, test_ids.numpy(), 'test-{}'.format(k))
            print('test finished, check {} for the results'.format(test_dir))


if __name__ == '__main__':
    main()
