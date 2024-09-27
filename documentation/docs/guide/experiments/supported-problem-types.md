# Supported problem types

## Overview

H2O LLM Studio supports various problem types that allow users to fine-tune models for different tasks. The five supported problem types are explained below.

## Causal language modeling 

- **Description:** Causal language modeling involves predicting the next token in a sequence, based only on the preceding tokens (i.e., the left side of the sequence). It is commonly used for tasks such as text generation. It is used to fine-tune large language models.

## Causal classification modeling

- **Description:** Causal classification modeling involves assigning one or more categorical target labels to an input text. It is used for fine-tuning models to perform text classification tasks.

- **Supported classification tasks:** Binary, multi-class, and multi-label classification.

## Causal regression modeling

- **Description:** Causal regression modeling assigns one or more continuous target labels to an input text. It is used to fine-tune models for text regression tasks. 

- **Supported regression tasks:** Multi-label regression.

## Sequence to sequence modeling

- **Description:** A type of machine learning architecture designed to transform one sequence into another. It is commonly used for tasks like machine translation, text summarization, and speech recognition.

## DPO modeling

- **Description:** The DPO modeling is used to fine-tune large language models using Direct Preference Optimization (DPO), a method that helps large, unsupervised language models better match human preferences using a simple classification approach. 

