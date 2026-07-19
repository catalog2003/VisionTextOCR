# VisionTextOCR

End-to-End Optical Character Recognition (OCR) using Swin Transformer Encoder and BART Decoder.

This project implements a transformer-based OCR system capable of recognizing:

- Scene Text
- Document Text
- Receipt Text
- Numbers
- Mixed Alphanumeric Text
- Long Sentences

Unlike traditional OCR pipelines that separate text detection and recognition, this project focuses on sequence generation using a Vision Encoder–Decoder architecture.

---

## Features

- Swin Transformer Vision Encoder
- BART Language Decoder
- Hugging Face VisionEncoderDecoderModel
- Mixed Precision (AMP)
- Distributed Data Parallel (DDP) Training
- Automatic Checkpointing
- Best Model Saving
- CER / WER / Exact Match Evaluation
- Beam Search Decoding
- Supports Variable-Length Text Recognition

---

## Model Architecture

Image
        │
        ▼
Swin Transformer Encoder
        │
        ▼
Vision Features
        │
        ▼
BART Decoder
        │
        ▼
Predicted Text

---

## Dataset

Training Dataset:

- IIIT5K
- Scene Text OCR Dataset
- Hugging Face OCR datasets

Images are automatically processed using:

- AutoImageProcessor
- AutoTokenizer

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Torch Distributed
- Mixed Precision Training
- CUDA
- Swin Transformer
- BART

---

## Training

Example:

```bash
python train_ocr.py
```

Distributed:

```bash
torchrun --nproc_per_node=2 train_ocr.py
```

---

## Evaluation Metrics

The model is evaluated using:

- Character Error Rate (CER)
- Word Error Rate (WER)
- Exact Match Accuracy
- Inference Latency
- Throughput

Example Results

| Metric | Score |
|---------|--------|
| CER Accuracy | **76.40%** |
| CER | **23.60%** |
| WER | **32.34%** |
| Exact Match Accuracy | **80.08%** |
| Average Latency | **122.75 ms/image** |
| Throughput | **16.29 images/sec** |

---

## Sample Predictions

| Ground Truth | Prediction |
|--------------|------------|
| DOLE | DOLE |
| SALE | SALE |
| MBA | MBA |
| Hauteur | Hauteur |
| 40.99 | 40.99 |
| TOTAL SALES INCLUSIVE GST @6% | TOTAL SALES INCLUSIVE GST @6% |

---

## Repository Structure

```
VisionTextOCR/
│
├── train_ocr.py
├── inference.py
├── checkpoints/
├── final_ocr_model/
├── best_ocr_model/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Future Improvements

- TrOCR Backbone
- Donut OCR
- Vision Transformer Encoder
- Synthetic OCR Pretraining
- Quantization
- ONNX Export
- TensorRT Optimization
- Web Demo using Gradio

---

## Performance

- 183.8 Million Parameters
- Transformer-based Encoder-Decoder OCR
- Beam Search Decoding
- Mixed Precision Training
- Multi-GPU Distributed Training

---

## Installation

```bash
git clone https://github.com/catalog2003/VisionTextOCR.git

cd VisionTextOCR

pip install -r requirements.txt
```

---

## License

MIT License

---

## Author



Computer Vision | Deep Learning | NLP | Generative AI
