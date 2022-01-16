# Formality-augmented Machine Translation

## Introduction

### Motivation

This project aims at modifying existing machine translation architectures so that they would be able to classify sentences according to their formality and reproduce sentences in the correct formality class.

### Problem description

We picked Japanese and Korean as our primary target languages, due to formality manifesting in their respective morphology. In addition to basic machine translation, our model performs two additional tasks:

- Indentify the formality class of the input sentence;

- Emit sentences in the target language with the correct formality class.

## Training data

Please contact the owners of this repository for details of and/or access to data that was used for this project.

### Formality labels

Bilingual corpora annotated with sentence formality are basically non-existent. As such, we devised several ways to generated formality labels from the corpora that were available to us.

#### Procedural generation of Japanese formality labels

We adapted an earlier work which classified the formality of a Japanese sentence according to the final verb of the sentence. The SOV sentence structure of Japanese means that the final verb is the outermost verb in any verb phrase embedding structure, and therefore also the main verb of the sentence. The classifier script can be found in formality_classification.py.

Two examples of sentence formality classifications:

- 同情してただけなんだ: informal

- 別に驚くことではないですよね: formal

For more details on rule-based formality classification, please refer to earlier work by Weston Feely, Eva Hasler and Adrià de Gispert. Their original repository can be found at <https://github.com/wfeely/japanese-verb-conjugator>.

### Using pre-trained Japanese language model for formality classification

We adapted the pre-trained Japanese BERT (credit to Tohoku University, available on huggingface) to output a politeness label given an input sentence. Script can be found at src/pre-trained.py

## Model Architecture

Base translation model is an autoregressive transformer. Model implementation can be found at src/model.py.

Training and evaluation scripts for our model can be found in the src directory.

## Results and bibliography

Please refer to our report at <https://hstehstehste.github.io/Projects/multi_formality_domain.html> for details on our experiment design, test results, and complete bibliography.
