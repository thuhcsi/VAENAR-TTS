import os
import threading
import multiprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from audio import Audio


class TestUtils:
    def __init__(self, hps, save_dir):
        self.prcocessor = Audio(hps.Audio)
        self.hps = hps
        self.save_dir = save_dir

    def write_mels(self, step, mel_batch, mel_lengths, ids, prefix=''):
        for i in range(mel_batch.shape[0]):
            mel = mel_batch[i][:mel_lengths[i], :]
            idx = ids[i].decode('utf-8') if type(ids[i]) is bytes else ids[i]
            mel_name = os.path.join(self.save_dir, '{}-{}-{}.npy'.format(prefix, idx, step))
            np.save(mel_name, mel)
        return

    def synthesize_and_save_wavs(self, step, mel_batch, mel_lengths, ids, prefix=''):
        def _synthesize(mel, fid):
            wav_arr = self.prcocessor.inv_mel_spectrogram(mel.T)
            wav_arr = self.prcocessor.inv_preemphasize(wav_arr)
            self.prcocessor.save_wav(wav_arr, os.path.join(self.save_dir, '{}-{}-{}.wav'.format(prefix, fid, step)))
            return
        threads = []
        for i in range(mel_batch.shape[0]):
            mel = mel_batch[i][:mel_lengths[i], :]
            idx = ids[i].decode('utf-8') if type(ids[i]) is bytes else ids[i]
            t = threading.Thread(target=_synthesize, args=(mel, idx))
            threads.append(t)
            t.start()
        for x in threads:
            x.join()
        print('All wavs for {} are synthesized!'.format(prefix))
        return

    @staticmethod
    def draw_mel_process(args):
        mel, ml, save_name = args
        plt.imshow(mel[:ml, :].T, aspect='auto', origin='lower')
        plt.tight_layout()
        plt.savefig('{}'.format(save_name))
        plt.close()

    def draw_melspectrograms(self, step, mel_batch, mel_lengths, ids, prefix=''):
        matplotlib.use('agg')
        save_names = []
        for idx in ids:
            idx = idx.decode('utf-8') if type(idx) is bytes else idx
            save_name = self.save_dir + '/{}-{}-{}.pdf'.format(prefix, idx, step)
            save_names.append(save_name)
        pool = multiprocessing.Pool()
        data = zip(mel_batch, mel_lengths, save_names)
        pool.map(TestUtils.draw_mel_process, data)
        return

    def _ids_to_symbols(self, id_list):
        # _pad = '_'
        # _eos = '~'
        # _characters = 'abcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '
        # symbols = [_pad, _eos] + list(_characters)
        _characters = self.hps.Texts.characters
        symbols = list(_characters)
        _id_to_symbol = {i: s for i, s in enumerate(symbols)}
        return [_id_to_symbol[x] for x in id_list]

    def draw_multi_head_att_process(self, args):
        ali, txt, tlen, mlen, save_name, num_heads = args
        txt = self._ids_to_symbols(txt)
        fig = plt.figure()
        for j, head_ali in enumerate(ali):
            ax = fig.add_subplot(2, num_heads // 2, j + 1)
            x = np.arange(tlen)
            ax.set_xticks(x)
            ax.set_xticklabels(txt[:tlen], fontsize=2)
            ax.imshow(head_ali[:, :tlen], aspect='auto', origin='lower')
        plt.tight_layout()
        plt.savefig('{}'.format(save_name))
        plt.close()
        return

    def draw_normal_att_process(self, args):
        ali, txt, tlen, mlen, save_name = args
        txt = self._ids_to_symbols(txt)
        x = np.arange(tlen)
        fig, ax = plt.subplots()
        ax.set_xticks(x)
        ax.set_xticklabels(txt[:tlen], fontsize=3)
        ax.imshow(ali[:mlen, : tlen], aspect='auto', origin='lower')
        plt.tight_layout()
        plt.savefig('{}.pdf'.format(save_name))
        plt.close()
        return

    def multi_draw_attention_alignments(self, batch_ali, batch_texts, text_lengths, mel_lengths, step, ids, prefix='posterior'):
        matplotlib.use('agg')
        save_names = []
        for idx in ids:
            idx = idx.decode('utf-8') if type(idx) is bytes else idx
            save_name = self.save_dir + '/{}-{}-{}.pdf'.format(prefix, idx, step)
            save_names.append(save_name)
        pool = multiprocessing.Pool()
        if len(batch_ali.shape) == 3:
            data = zip(batch_ali, batch_texts, text_lengths, mel_lengths, save_names)
            pool.map(self.draw_normal_att_process, data)
        elif len(batch_ali.shape) == 4:
            data = zip(batch_ali, batch_texts, text_lengths, mel_lengths, save_names,
                       [batch_ali.shape[1]] * batch_ali.shape[0])
            pool.map(self.draw_multi_head_att_process, data)
        print('Attentions for {} are plotted'.format(prefix))
        return
