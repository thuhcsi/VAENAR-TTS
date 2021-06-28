import tensorflow as tf
import numpy as np
import argparse
import time
import os

from tqdm import tqdm
from configs import LJHPS, DataBakerHPS
from audio import TestUtils
from datasets import TFRecordWriter, LJSpeech, DataBaker
from models import VAENAR


def synthesize_from_text():
    parser = argparse.ArgumentParser('Training parameters parser')
    parser.add_argument('--dataset', type=str, choices=['ljspeech', 'databaker', 'cantonese'],
                        help='dataset name, currently support ljspeech and databaker')
    parser.add_argument('--text', type=str,
                        help='text file contains multiple lines of text to be synthesized')
    parser.add_argument('--ckpt_path', type=str,
                        help='path to the model ckpt')
    parser.add_argument('--test_dir', type=str,
                        help='directory to save test results')
    parser.add_argument('--temperature', type=float, default=0.)
    args = parser.parse_args()
    # validate the paths
    ckpt_path = args.ckpt_path
    ckpt_step = ckpt_path.split('-')[-1]
    assert os.path.isfile(args.text)
    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    hparams = {'ljspeech': LJHPS, 'databaker': DataBakerHPS}[args.dataset]
    dataset = {'ljspeech': LJSpeech, 'databaker': DataBaker}[args.dataset](
        data_root=None, save_dir=None, hps=hparams)
    tester = TestUtils(hparams, args.test_dir)
    # setup model
    model = VAENAR(hparams)
    checkpoint = tf.train.Checkpoint(model=model)
    # model.load_weights(ckpt_path)
    checkpoint.restore(ckpt_path).expect_partial()
    # prediction
    text_lens = []
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            line = line.strip()
            text = dataset.text_to_array(line)
            text_lens.append(len(text))
            texts.append(text)
    ids = [str(i) for i in range(len(text_lens))]
    text_max_len = np.max(text_lens)
    text_batch = np.stack([t + (text_max_len - len(t)) * [0] for t in texts], axis=0)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def test_step(t, t_l):
        text_pos_step = model.mel_text_len_ratio / tf.cast(
            hparams.Common.final_reduction_factor, tf.float32)
        text_embd = model.text_encoder(t, t_l, pos_step=text_pos_step, training=False)
        text_embd.set_shape([None, None, hparams.Encoder.Transformer.embd_dim])
        predicted_lengths = model.length_predictor(
            tf.stop_gradient(text_embd), t_l, training=False)
        predicted_m_l = tf.cast(predicted_lengths, tf.int32)
        reduced_pred_ml = (predicted_m_l + 80 + hparams.Common.final_reduction_factor - 1
                           ) // hparams.Common.final_reduction_factor
        prior_latents, prior_logprobs = model.prior.sample(
            reduced_pred_ml, text_embd, t_l, training=False, temperature=args.temperature)
        _, prior_dec_outs, prior_dec_alignments = model.decoder(
            prior_latents, text_embd, reduced_pred_ml, t_l, training=False)
        return prior_dec_outs, predicted_m_l + 80, prior_dec_alignments

    prediction, pred_lens, dec_alignments = test_step(tf.constant(text_batch, dtype=tf.int32),
                                                      tf.constant(text_lens, dtype=tf.int32))
    tester.synthesize_and_save_wavs(ckpt_step, prediction.numpy(), pred_lens.numpy(), ids, prefix='test')
    for k in dec_alignments.keys():
        tester.multi_draw_attention_alignments(
            dec_alignments[k].numpy(), texts, text_lens,
            pred_lens.numpy(), ckpt_step, ids, 'prior-{}'.format(k))
    return


def inference_test():
    parser = argparse.ArgumentParser('Training parameters parser')
    parser.add_argument('--dataset', type=str, choices=['ljspeech', 'databaker', 'cantonese'],
                        help='dataset name, currently support ljspeech, databaker and cantonese')
    parser.add_argument('--data_dir', type=str,
                        help='dataset root directory')
    parser.add_argument('--ckpt_path', type=str,
                        help='path to the model ckpt')
    parser.add_argument('--test_dir', type=str,
                        help='directory to save test results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--write_mels', type=bool, default=True)
    parser.add_argument('--write_wavs', type=bool, default=False)
    parser.add_argument('--draw_alignments', type=bool, default=False)
    args = parser.parse_args()
    # validate the paths
    ckpt_path = args.ckpt_path
    ckpt_step = ckpt_path.split('-')[-1]
    test_dir = args.test_dir
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    # setup hparams
    hparams = {'ljspeech': LJHPS, 'databaker': DataBakerHPS}[args.dataset]
    tester = TestUtils(hparams, args.test_dir)
    # 1. loading dataset
    data_records = TFRecordWriter(save_dir=args.data_dir)
    test_set = data_records.create_dataset(
        buffer_size=hparams.Dataset.buffer_size,
        num_parallel_reads=hparams.Dataset.num_parallel_reads,
        pad_factor=hparams.Dataset.pad_factor,
        batch_size=args.batch_size,
        num_mels=hparams.Audio.num_mels,
        shuffle_buffer=hparams.Train.shuffle_buffer,
        shuffle=hparams.Train.shuffle,
        tfrecord_files=data_records.get_tfrecords_list('test'))
    # setup model
    model = VAENAR(hparams)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path).expect_partial()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def test_step(t, t_l):
        text_pos_step = model.mel_text_len_ratio / tf.cast(
            hparams.Common.final_reduction_factor, tf.float32)
        text_embd = model.text_encoder(t, t_l, pos_step=text_pos_step, training=False)
        text_embd.set_shape([None, None, hparams.Encoder.Transformer.embd_dim])
        predicted_lengths = model.length_predictor(
            tf.stop_gradient(text_embd), t_l, training=False)
        predicted_m_l = tf.cast(predicted_lengths, tf.int32)
        reduced_pred_ml = (predicted_m_l + 80 + hparams.Common.final_reduction_factor - 1
                           ) // hparams.Common.final_reduction_factor
        prior_latents, prior_logprobs = model.prior.sample(
            reduced_pred_ml, text_embd, t_l, training=False, temperature=args.temperature)
        _, prior_dec_outs, prior_dec_alignments = model.decoder(
            prior_latents, text_embd, reduced_pred_ml, t_l, training=False,
            reduction_factor=hparams.Common.final_reduction_factor)
        return prior_dec_outs, predicted_m_l + 80, prior_dec_alignments

    # tf.function initialization
    for _, texts, _, t_lengths, _ in test_set.take(1):
        _, _, _ = test_step(texts, t_lengths)
    time_consumed = 0.
    durations = 0.
    for fids, texts, _, t_lengths, _ in tqdm(test_set):
        time_begin = time.time()
        prior_outs, pred_m_lens, prior_ali = test_step(texts, t_lengths)
        time_end = time.time()
        time_consumed += time_end - time_begin
        durations += np.sum(pred_m_lens.numpy()) * hparams.Audio.frame_shift_sample / hparams.Audio.sample_rate
        if args.write_mels:
            tester.write_mels(ckpt_step, prior_outs.numpy(), pred_m_lens.numpy(), fids.numpy(), prefix='prior')
        if args.write_wavs:
            tester.synthesize_and_save_wavs(ckpt_step, prior_outs.numpy(), pred_m_lens.numpy(), fids.numpy(), prefix='prior')
        if args.draw_alignments:
            for k in prior_ali.keys():
                tester.multi_draw_attention_alignments(
                    prior_ali[k].numpy(), texts.numpy(), t_lengths.numpy(),
                    pred_m_lens.numpy(), ckpt_step, fids.numpy(), 'prior-{}'.format(k))
    average_rtf = time_consumed / durations
    print('Total time consumed is {} Secs,'
          'total synthesis duration is {} Secs,'
          'Average RTF is {}.'.format(time_consumed, durations, average_rtf))


if __name__ == '__main__':
    inference_test()
    # synthesize_from_text()
