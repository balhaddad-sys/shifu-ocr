#!/usr/bin/env python3
"""
Fine-tune PaddleOCR recognition model on MedTriage medical text.
Runs on CPU — no GPU required.

This trains the PP-OCRv3 English recognition model on our custom
medical vocabulary (ward sheets, diagnoses, medications, names).

Usage:
    python finetune_rec.py [--epochs 5] [--batch 16]
    python finetune_rec.py --export  # Export trained model to ONNX
"""
import os
import sys
import json
import shutil
import subprocess

BASE = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE, 'training_data')
MODEL_DIR = os.path.join(BASE, 'pretrained')
OUTPUT_DIR = os.path.join(BASE, 'trained_model')
PADDLEOCR_DIR = os.path.join(BASE, 'PaddleOCR')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_prerequisites():
    """Verify training data and tools exist."""
    train_list = os.path.join(TRAIN_DIR, 'train_list.txt')
    val_list = os.path.join(TRAIN_DIR, 'val_list.txt')
    dict_path = os.path.join(TRAIN_DIR, 'med_dict.txt')

    if not os.path.exists(train_list):
        print('ERROR: Training data not found. Run prepare_training.py first.')
        return False
    if not os.path.exists(dict_path):
        print('ERROR: Dictionary not found. Run prepare_training.py first.')
        return False

    with open(train_list) as f:
        train_count = sum(1 for _ in f)
    with open(val_list) as f:
        val_count = sum(1 for _ in f)
    with open(dict_path) as f:
        dict_size = sum(1 for _ in f)

    print(f'Training data: {train_count} samples')
    print(f'Validation data: {val_count} samples')
    print(f'Dictionary: {dict_size} characters')
    return True


def clone_paddleocr():
    """Clone PaddleOCR repo for training tools."""
    if os.path.exists(PADDLEOCR_DIR):
        print('PaddleOCR already cloned.')
        return True

    print('Cloning PaddleOCR (for training tools)...')
    result = subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/PaddlePaddle/PaddleOCR.git', PADDLEOCR_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f'ERROR: git clone failed: {result.stderr}')
        return False
    print('PaddleOCR cloned successfully.')
    return True


def download_pretrained():
    """Download PP-OCRv3 English recognition pretrained model."""
    model_url = 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar'
    model_tar = os.path.join(MODEL_DIR, 'en_PP-OCRv3_rec_train.tar')
    model_extracted = os.path.join(MODEL_DIR, 'en_PP-OCRv3_rec_train')

    if os.path.exists(model_extracted):
        print('Pretrained model already downloaded.')
        return model_extracted

    print('Downloading PP-OCRv3 English recognition model...')
    import urllib.request
    urllib.request.urlretrieve(model_url, model_tar)

    print('Extracting...')
    import tarfile
    with tarfile.open(model_tar, 'r') as tar:
        tar.extractall(MODEL_DIR)

    if os.path.exists(model_tar):
        os.remove(model_tar)

    print(f'Pretrained model ready: {model_extracted}')
    return model_extracted


def write_training_config(pretrained_path, epochs=5, batch_size=16):
    """Generate PaddleOCR training configuration YAML."""
    config = f"""Global:
  debug: false
  use_gpu: false
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 50
  save_model_dir: {OUTPUT_DIR}
  save_epoch_step: 1
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: {pretrained_path}/best_accuracy
  checkpoints:
  use_visualdl: false
  save_inference_dir:
  character_dict_path: {TRAIN_DIR}/med_dict.txt
  character_type: en
  max_text_length: 80
  infer_mode: false
  use_space_char: true
  save_res_path: {OUTPUT_DIR}/rec_results.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: true
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 80

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {TRAIN_DIR}
    label_file_list:
      - {TRAIN_DIR}/train_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_sar
            - length
            - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: {batch_size}
    drop_last: true
    num_workers: 0

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {TRAIN_DIR}
    label_file_list:
      - {TRAIN_DIR}/val_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_sar
            - length
            - valid_ratio
  loader:
    shuffle: false
    batch_size_per_card: {batch_size}
    drop_last: false
    num_workers: 0
"""
    config_path = os.path.join(TRAIN_DIR, 'med_rec_config.yml')
    with open(config_path, 'w') as f:
        f.write(config)
    print(f'Training config written: {config_path}')
    return config_path


def run_training(config_path):
    """Execute PaddleOCR training."""
    train_script = os.path.join(PADDLEOCR_DIR, 'tools', 'train.py')
    if not os.path.exists(train_script):
        print(f'ERROR: PaddleOCR train.py not found at {train_script}')
        print('Run with --clone first to download PaddleOCR tools.')
        return False

    cmd = [sys.executable, train_script, '-c', config_path]
    print(f'\nStarting training: {" ".join(cmd)}')
    print('This will take a while on CPU. Monitor output below.\n')

    result = subprocess.run(cmd, cwd=PADDLEOCR_DIR)
    return result.returncode == 0


def export_to_onnx():
    """Export trained model to ONNX for browser deployment."""
    export_script = os.path.join(PADDLEOCR_DIR, 'tools', 'export_model.py')
    config_path = os.path.join(TRAIN_DIR, 'med_rec_config.yml')
    inference_dir = os.path.join(OUTPUT_DIR, 'inference')

    if not os.path.exists(export_script):
        print('ERROR: PaddleOCR export tool not found.')
        return False

    cmd = [
        sys.executable, export_script,
        '-c', config_path,
        '-o', f'Global.pretrained_model={OUTPUT_DIR}/best_accuracy',
        '-o', f'Global.save_inference_dir={inference_dir}',
    ]

    print(f'Exporting model: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=PADDLEOCR_DIR)

    if result.returncode == 0:
        print(f'\nInference model exported to: {inference_dir}')
        print('Next step: Convert to ONNX with paddle2onnx for browser deployment.')
    return result.returncode == 0


def main():
    epochs = 5
    batch_size = 16

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--epochs' and i < len(sys.argv) - 1:
            epochs = int(sys.argv[i + 1])
        if arg == '--batch' and i < len(sys.argv) - 1:
            batch_size = int(sys.argv[i + 1])

    if '--export' in sys.argv:
        export_to_onnx()
        return

    print('=== MedTriage OCR Fine-Tuning (CPU) ===\n')

    # 1. Check training data
    if not check_prerequisites():
        return

    # 2. Clone PaddleOCR tools
    if not clone_paddleocr():
        return

    # 3. Download pretrained model
    pretrained = download_pretrained()

    # 4. Write config
    config_path = write_training_config(pretrained, epochs, batch_size)

    # 5. Train
    print(f'\nTraining for {epochs} epochs with batch size {batch_size}...')
    print('(CPU training — expect ~30-60 min per epoch with 9000 samples)\n')
    success = run_training(config_path)

    if success:
        print('\n=== TRAINING COMPLETE ===')
        print(f'Model saved to: {OUTPUT_DIR}')
        print('Run with --export to convert for browser deployment.')
    else:
        print('\nTraining failed. Check output above for errors.')


if __name__ == '__main__':
    main()
