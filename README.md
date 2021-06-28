# VAENAR-TTS
This repo contains code accompanying the paper "VAENAR-TTS: Variational Auto-Encoder based Non-AutoRegressive Text-to-Speech Synthesis".

## Samples are [here](https://light1726.github.io/vaenar-tts/)

## Usage

### 0. Dataset
1. English: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
2. Mandarin: [DataBaker(标贝)](https://www.data-baker.com/data/index/source/)

### 1. Environment setup
```bash
conda env create -f environment.yml
conda activate vaenartts-env
```

### 2. Data pre-processing

For English using LJSpeech:
```bash
CUDA_VISIBLE_DEVICES= python preprocess.py --dataset ljspeech --data_dir /path/to/extracted/LJSpeech-1.1 --save_dir ./ljspeech
```
For Mandarin using Databaker(标贝):
```bash
CUDA_VISIBLE_DEVICES= python preprocess.py --dataset databaker --data_dir /path/to/extracted/biaobei --save_dir ./databaker
```

### 3. Training
For English using LJSpeech:
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python train.py --dataset ljspeech --log_dir ./lj-log_dir --test_dir ./lj-test_dir --data_dir ./ljspeech/tfrecords/ --model_dir ./lj-model_dir
```
For Mandarin using Databaker(标贝):
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python train.py --dataset databaker --log_dir ./db-log_dir --test_dir ./db-test_dir --data_dir ./databaker/tfrecords/ --model_dir ./db-model_dir
```

### 4. Inference (synthesize speech for the whole test set)
For English using LJSpeech:
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python inference.py --dataset ljspeech --test_dir ./lj-test-2000 --data_dir ./ljspeech/tfrecords/ --batch_size 16 --write_wavs true --draw_alignments true --ckpt_path ./lj-model_dir/ckpt-2000
```
For Mandarin using Databaker(标贝):
```bash
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python inference.py --dataset databaker --test_dir ./db-test-2000 --data_dir ./databaker/tfrecords/ --batch_size 16 --write_wavs true --draw_alignments true --ckpt_path ./db-model_dir/ckpt-2000
```

## Reference
1. [XuezheMax/flowseq](https://github.com/XuezheMax/flowseq)
2. [keithito/tacotron](https://github.com/keithito/tacotron)