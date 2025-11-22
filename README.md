# Fuzzing Pipeline (Omission Detection)

This README presents the workflow of the NMTFuzz, which takes the H-NLP
translation model as an example to detect omission errors.

The core idea:

1. Slightly mutate an English source sentence.
2. Translate the mutated sentence with the H-NLP model.
3. Run word alignment (awesome-align) between English and the translation.
4. If many non-stopword English tokens are unaligned, treat it as a
   potential omission bug.
5. Use a token-vs-EOS probability metric as a testing guidance to decide
   which mutants to explore further.

## 1. Dependencies

Main Python libraries:

- `torch`
- `transformers`
  - `MarianMTModel`, `MarianTokenizer` (Helsinki model)
  - `BertForMaskedLM`, `BertTokenizer` (for masked-LM mutation)
- `nltk` (stopwords)
- Local modules:
  - `calculate_rouge1.cal_rouge1` – ROUGE-1 similarity
  - `mutation` – three mutation operators:
    - `mutate_word_character`
    - `mutate_punctuations`
    - `predict_masked_word_by_position`
  - `word_align` & `word_align.awesome_align` – alignment model & utilities
  - `word_align.data_tokenize.transform` – prepare sentence pair for awesome-align

