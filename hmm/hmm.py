import itertools
import json
import os
import warnings

warnings.filterwarnings("ignore")

from typing import List

import numpy as np
import pandas as pd

from utils import VocabConfig, HMMConfig


class HMM:
    def __init__(self, vocab_file: str, labels: List[str]):
        """Initialize the HMM model

        Args:
            vocab_file (str): Path to the vocab file
            labels (List[str]): List of tags
        """
        self.vocab = self._read_vocab(vocab_file)
        self.labels = labels

        # Hidden Markov Model Parameters
        self.states = list()
        self.priors = None
        self.transitions = None
        self.emissions = None

        # Laplace Smoothing
        self.smoothing_constant = 1e-10

    def _read_vocab(self, vocab_file: str):
        return pd.read_csv(vocab_file, sep="\t", names=VocabConfig.FILE_HEADER)

    def _initialize_params(self):
        self.states = list(self.labels)

        # N = Number of states i.e. number of distinct tags
        num_states = len(self.labels)
        # M = Number of observable symbols i.e. number of distinct words
        num_observations = len(self.vocab)

        # State transition probability matrix of size N * N
        self.transitions = np.zeros((num_states, num_states))

        # Obseravtion Emission probability matrix of size N * M
        self.emissions = np.zeros((num_states, num_observations))

        # Prior probability matrix of size N * 1
        self.priors = np.zeros(num_states)

    def _smoothen_propabilities(self, prob_mat: np.array, smoothing_constant: float):
        """Handle cases where the probabilities is 0"""
        return np.where(prob_mat == 0, smoothing_constant, prob_mat)

    def _compute_prior_params(self, train_data):
        """Compute the prior probabilities

        Formula: π(s) = count(null -> s) / count(num_sentences)
        """
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}
        num_sentences = len(train_data)

        for sentence in train_data:
            label = sentence[0][1]
            state_idx = tag_to_index[label]
            self.priors[state_idx] += 1

        self.priors = self.priors / num_sentences
        self.priors = self._smoothen_propabilities(self.priors, self.smoothing_constant)

    def _compute_transition_params(self, train_data):
        """Compute transition parameters

        Formula: t(s′|s) = count(s -> s′) / count(s)
        """
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}

        for sentence in train_data:
            label_indices = [tag_to_index.get(label) for _, label in sentence]

            for i in range(1, len(label_indices)):
                prev_state = label_indices[i - 1]
                curr_state = label_indices[i]
                self.transitions[prev_state, curr_state] += 1

        row_agg = self.transitions.sum(axis=1)[:, np.newaxis]
        self.transitions = self.transitions / row_agg
        self.transitions = self._smoothen_propabilities(self.transitions, self.smoothing_constant)

    def _compute_emission_params(self, train_data):
        """Compute emission parameters

        Formula: e(x|s) = count(s -> x) / count(s)
        """
        word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}

        for sentence in train_data:
            for word, label in sentence:
                state_idx = tag_to_index[label]
                word_idx = word_to_index.get(word, word_to_index[VocabConfig.UNKNOWN_TOKEN])
                self.emissions[state_idx, word_idx] += 1

        row_agg = self.emissions.sum(axis=1)[:, np.newaxis]
        self.emissions = self.emissions / row_agg
        self.emissions = self._smoothen_propabilities(self.emissions, self.smoothing_constant)

    def fit(self, train_data: pd.DataFrame):
        self._initialize_params()
        self._compute_prior_params(train_data)
        self._compute_transition_params(train_data)
        self._compute_emission_params(train_data)

    @property
    def get_all_probability_matrices(self):
        return self.priors, self.transitions, self.emissions

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = HMMConfig.HMM_MODEL_SAVED

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        transition_prob = {
            f"({s1}, {s2})": self.transitions[self.states.index(s1), self.states.index(s2)]
            for s1, s2 in itertools.product(self.states, repeat=2)
        }

        emission_prob = {
            f"({s}, {w})": p
            for s in self.states
            for w, p in zip(self.vocab["word"], self.emissions[self.states.index(s), :])
        }

        model_params = {"transition": transition_prob, "emission": emission_prob}

        with open(file_path, "w") as json_file:
            json.dump(model_params, json_file, indent=4)

        print(f"Saving model to {os.path.relpath(file_path)}")


class AdversarialHMM(HMM):
    def __init__(self, vocab_file: str, labels: List[str]):
        super().__init__(vocab_file, labels)

    def _perturb_emission_params(self):
        """Introduce perturbations in emission parameters"""
        # Generate perturbation matching the shape of the emission matrix
        perturbation = np.random.normal(loc=0, scale=0.001, size=self.emissions.shape)

        # Add perturbation to emission probabilities
        self.emissions += perturbation

    def fit(self, train_data: pd.DataFrame):
        self._initialize_params()
        self._compute_prior_params(train_data)
        self._compute_transition_params(train_data)
        self._compute_emission_params(train_data)
        self._perturb_emission_params()
