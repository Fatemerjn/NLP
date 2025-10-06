# NLP Movie Genre Classification

This repository focuses on classifying movie genres from text such as plot summaries or descriptions using natural language processing (NLP) techniques. Several approaches are implemented, ranging from baseline document-level classifiers to transformer-based models and sequence labeling architectures. The work covers both **English** and **Persian** datasets.

---

## Repository Contents

- `NLP_HW3_Bert_Persian.ipynb` – fine-tuning BERT for Persian movie genre classification  
- `NLP_HW3_Doc_Base_Persian.ipynb` – a baseline document-level classifier for Persian texts  
- `NLP_HW3_NER_LSTM_CRF_English.ipynb` – English experiment combining NER, LSTM, and CRF  
- `NLP_HW3_NER_LSTM_CRF_Persian.ipynb` – Persian experiment with NER + LSTM + CRF pipeline  
- `dataset.csv` / `dataset.xlsx` – datasets containing movie descriptions and associated genre labels  

---

## Motivation

Most systems assign genres to movies using metadata such as tags or user annotations. The goal here is to build a system that can infer genres directly from the **textual content** of movie summaries. This approach is especially useful when metadata is incomplete, inconsistent, or unavailable.  

By experimenting with multilingual data (English and Persian) and multiple architectures, this project explores how well models can generalize across languages and classification setups.  

---

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Fatemerjn/NLP-Movie-Genre-Classification-.git
   cd NLP-Movie-Genre-Classification-
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers pandas scikit-learn seqeval
   ```

3. Open one of the Jupyter notebooks and run step by step:
   - Start with `NLP_HW3_Doc_Base_Persian.ipynb` for a baseline  
   - Explore `NLP_HW3_Bert_Persian.ipynb` for transformer-based classification  
   - Use the NER + LSTM + CRF notebooks for sequence labeling approaches  

4. Make sure to adjust dataset paths if needed.  

---

## Dataset

The dataset files (`dataset.csv`, `dataset.xlsx`) contain movie descriptions with associated genres. Some preprocessing and label encoding steps are performed in the notebooks before training. Class imbalance may be present and should be considered during evaluation.  

---

## Key Experiments

- **Document-level classification** with traditional and neural baselines  
- **Transformer-based models (BERT)** fine-tuned for Persian  
- **NER + LSTM + CRF pipelines** for extracting features and predicting genres  
- **Cross-language exploration** between English and Persian data  

---

## Evaluation

The notebooks demonstrate use of common metrics including:  

- Accuracy  
- Precision, Recall, and F1-score  
- Sequence labeling metrics for NER (via `seqeval`)  

---

## Suggestions for Extension

- Expand the dataset with more movies and genres  
- Explore cross-lingual transfer (training on one language, testing on another)  
- Incorporate multilingual transformers such as XLM-R or mT5  
- Experiment with multi-label classification when movies belong to multiple genres  
- Add interpretability methods to highlight important words influencing predictions  

---

## References

- Hugging Face Transformers library for pretrained models  
- PyTorch for model training and evaluation  
- Scikit-learn for preprocessing and baseline classifiers  
- Seqeval for sequence labeling evaluation  

---

## License

This repository is intended for educational and research purposes. 
