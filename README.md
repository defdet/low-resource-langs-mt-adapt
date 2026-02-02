# LoResMT 2026 Shared Task Submission

This repository contains the codebase for our LoResMT 2026 Shared Task submission. It includes training scripts, inference code, back-translation script, and links to  adapters's weights hosted on HuggingFace Hub.

## Training

Training is implemented via self-contained Jupyter notebooks:

- `tencent-gpu-mt-kazakh.ipynb` - Russian -> Kazakh translation
- `tencent-gpu-mt-bashkir.ipynb` - Russian -> Bashkir translation  
- `tencent-gpu-mt-all-chuvash.ipynb` - English -> Chuvash translation

## Inference and Back-Translation

- `tencent_gpu_gen.ipynb` - Generation notebook for submission files
- `translate_ru_to_en_and_push.py` - Back-translation script

## Model Weights

- [`Defetya/tencent-chuvash-7b-adapter`](https://huggingface.co/Defetya/tencent-chuvash-7b-adapter) - Chuvash LoRA adapters
- [`Defetya/tencent-kazakh-7b-adapter`](https://huggingface.co/Defetya/tencent-kazakh-7b-adapter) - Kazakh LoRA adapters
- [`Defetya/tencent-7b-bashkir-lora`](https://huggingface.co/Defetya/tencent-7b-bashkir-lora) - Bashkir LoRA adapters

## Datasets

- [`Defetya/chuvash_russian_english_parallel`](https://huggingface.co/datasets/Defetya/chuvash_russian_english_parallel) - Back-translated parallel corpus
