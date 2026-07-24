# English–Bangla Neural Machine Translation

A comparison of two approaches to Neural Machine Translation (NMT) for the English–Bangla language pair: a Transformer encoder-decoder trained entirely **from scratch**, versus a pretrained multilingual model (**NLLB-200**) adapted via **transfer learning (fine-tuning)**.

Final project for CMPE 252 – Artificial Intelligence and Data Engineering.

## Results

| Metric | From-Scratch Transformer | NLLB Fine-Tuned |
|---|---|---|
| BLEU score | 2.86 | **9.23** |
| Parameters | 10,106,688 | 615,073,792 |
| Training examples | 99,260 | 20,000 |
| Epochs | 25 | 2 |
| Final training loss | 2.59 | 1.65 |
| Hardware | Tesla T4 (free-tier Colab) | Tesla T4 (free-tier Colab) |

The fine-tuned NLLB model achieved a BLEU score **more than 3x higher** than the from-scratch model, despite using roughly one-fifth of the training data — demonstrating a clear, measurable advantage of transfer learning under constrained data and compute conditions.

## Project Structure

```
├── EN-BN-NMT-Project.ipynb        # Full notebook: data prep, both models, training, evaluation
├── EN-BN-NMT-Project-Report.docx  # Full written report (methodology, results, challenges)
├── spm_en.model / spm_en.vocab    # SentencePiece tokenizer (English)
├── spm_bn.model / spm_bn.vocab    # SentencePiece tokenizer (Bangla)
├── results_from_scratch.json      # Saved metrics for the from-scratch model
├── loss_curve_combined.png        # Training loss, from-scratch model (both runs)
├── nllb_loss_curve.png            # Training loss, NLLB fine-tuning
└── README.md
```

> **Note:** Trained model weights (`checkpoint.pt`, ~40MB, and the fine-tuned NLLB checkpoint, ~2.4GB) are not included in this repository due to GitHub file size limits. Both models can be reproduced by running the notebook end-to-end — see below.

## Dataset

[OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) English–Bangla subset (1M training pairs). Rows with corrupted Bengali virama spacing (~31% of the raw data) were filtered out, leaving 858,299 clean examples.

## Method Summary

**From-scratch model:** A Transformer encoder-decoder (3 encoder + 3 decoder layers, 4 attention heads, 256-dim embeddings) built with PyTorch's `nn.Transformer`, using custom SentencePiece tokenizers (8,000-token vocabulary per language, BPE) trained from scratch on the project data.

**Fine-tuned model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M), fine-tuned on a 20,000-example subset. Fitting this on a free-tier T4 GPU required gradient checkpointing, mixed-precision (fp16) training, and gradient accumulation (effective batch size 8, physical batch size 1).

Full methodology, hyperparameters, and a qualitative error analysis (including a documented token-repetition failure mode in the from-scratch model) are in the [project report](./EN-BN-NMT-Project-Report.docx).

## Reproducing This Project

1. Open `EN-BN-NMT-Project.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run cells from the top — the notebook downloads OPUS-100 automatically via Hugging Face `datasets`
4. Tokenizer files in this repo can be loaded directly (see the "load tokenizer" cell) instead of retraining them

**Requirements:** `datasets`, `sentencepiece`, `sacrebleu`, `transformers`, `torch` (installed via pip in the first notebook cell).

## Evaluation

Both models were evaluated with [sacreBLEU](https://github.com/mjpost/sacrebleu) (corpus-level BLEU) on the full, unseen OPUS-100 test set (2,000 sentence pairs).

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- NLLB Team, Meta AI. *No Language Left Behind: Scaling Human-Centered Machine Translation.*
- Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.*
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.*
