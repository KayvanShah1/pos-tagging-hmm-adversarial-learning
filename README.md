# Part-of-Speech Tagging using Hidden Markov Models

This repository contains the implementation of a Part-of-Speech (POS) tagging system using Hidden Markov Models (HMMs) along with various decoding techniques and adversarial training strategies for sequence labeling tasks.

## Overview

The project focuses on:
- **HMM-based POS Tagging:** Implementing HMMs for sequence labeling, associating hidden states (POS tags) with observed words.
- **Decoding Techniques Comparison:** Comparing and evaluating Greedy Decoding and Viterbi Decoding methods for sequence decoding.
- **Adversarial Training:** Exploring adversarial techniques to enhance the robustness and accuracy of the HMM-based POS tagging system.

## Key Components

- **Datasets:** Wall Street Journal Dataset is utilized for training, validation, and testing purposes.
- **Algorithms:** Implementation of HMMs, Greedy Decoding, Viterbi Decoding, and Adversarial HMMs.
- **Evaluation:** Calculating accuracy metrics for decoding techniques and adversarial training strategies.

## Installation and Usage

1. **Dependencies:** Ensure Python (version 3.10) and necessary libraries are installed.
2. **Dataset:** Sample of Wall Street Journal Dataset (provided in `/data`).
3. **Training:** Execute `train_and_evaluate_hmm()` and `train_and_evaluate_advs_hmm()` functions in `main.py` for respective models.
4. **Decoding:** Access decoding techniques via the `decoder.py` module.
5. **Evaluation:** Evaluate performance metrics using the provided evaluation functions.

> [!NOTE]
> For every use case comment out the part not required in the `hmm.py` script and run the `main.py` script.

## Course Information

- **Course:** DSCI 599 - Optimization Techniques for Data Science
- **Term:** Fall 2023
- **Project Title:** Part-of-Speech Tagging using Hidden Markov Models: Comparing decoding techniques and exploring adversarial training strategies

This project was developed as the final project for the Optimization Techniques for Data Science course offered in Fall 2023.
