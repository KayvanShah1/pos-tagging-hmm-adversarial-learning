import numpy as np

from .adv_hmm import VocabConfig


class GreedyDecoding:
    def __init__(self, prior_probs, transition_probs, emission_probs, states, vocab):
        self.priors = prior_probs
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.states = states
        self.vocab = vocab

        self.tag_to_idx = {tag: idx for idx, tag in enumerate(states)}
        self.word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))

        # Precompute scores for each word-tag pair
        self.priors_emissions = prior_probs[:, np.newaxis] * emission_probs

    def _decode_single_sentence(self, sentence):
        predicted_tags = []

        prev_tag_idx = None

        for word in sentence:
            word_idx = self.word_to_index.get(word, self.word_to_index[VocabConfig.UNKNOWN_TOKEN])

            if prev_tag_idx is None:
                # scores = self.priors * self.emissions[:, word_idx]
                scores = self.priors_emissions[:, word_idx]
            else:
                scores = self.transitions[prev_tag_idx] * self.emissions[:, word_idx]

            prev_tag_idx = np.argmax(scores)
            predicted_tags.append(self.states[prev_tag_idx])

        return predicted_tags

    def decode(self, sentences):
        predicted_tags_list = []

        for sentence in sentences:
            predicted_tags = self._decode_single_sentence([word for word, tag in sentence])
            predicted_tags_list.append(predicted_tags)

        return predicted_tags_list


class ViterbiDecoding:
    def __init__(self, prior_probs, transition_probs, emission_probs, states, vocab):
        self.priors = prior_probs
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.states = states
        self.vocab = vocab

        self.num_states = len(self.states)

        # Index Conversion dictionary for mapping
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(states)}
        self.word_to_idx = dict(zip(self.vocab["word"], self.vocab["index"]))

        # Precompute scores for each word-tag pair
        self.priors_emissions = prior_probs[:, np.newaxis] * emission_probs

    def _initialize_variables(self, sentence):
        V = np.zeros((len(sentence), self.num_states))
        path = np.zeros((len(sentence), self.num_states), dtype=int)

        word_idx = np.array(
            [
                self.word_to_idx.get(word, self.word_to_idx[VocabConfig.UNKNOWN_TOKEN])
                for word in sentence
            ]
        )

        return V, path, word_idx

    def _decode_single_sentence(self, sentence):
        V, path, word_idx = self._initialize_variables(sentence)

        V[0] = np.log(self.priors_emissions[:, word_idx[0]])

        for t in range(1, len(sentence)):
            # Compute scores
            scores = (
                V[t - 1, :, np.newaxis]
                + np.log(self.transitions)
                + np.log(self.emissions[:, word_idx[t]])
            )
            V[t] = np.max(scores, axis=0)
            path[t] = np.argmax(scores, axis=0)

        # Backtracking
        predicted_tags = [0] * len(sentence)
        predicted_tags[-1] = np.argmax(V[-1])

        for t in range(len(sentence) - 2, -1, -1):
            predicted_tags[t] = path[t + 1, predicted_tags[t + 1]]

        predicted_tags = [self.states[tag_idx] for tag_idx in predicted_tags]
        return predicted_tags

    def decode(self, sentences):
        predicted_tags_list = []

        for sentence in sentences:
            predicted_tags = self._decode_single_sentence([word for word, tag in sentence])
            predicted_tags_list.append(predicted_tags)

        return predicted_tags_list


def calculate_accuracy(predicted_sequences, true_sequences):
    """
    Calculate the accuracy of predicted sequences compared to true sequences.

    Args:
        predicted_sequences (list): List of predicted sequences.
        true_sequences (list): List of true sequences.

    Returns:
        float: Accuracy.
    """
    total = 0
    correct = 0

    for true_label, predicted_label in zip(true_sequences, predicted_sequences):
        for true_tag, predicted_tag in zip(true_label, predicted_label):
            total += 1
            if true_tag == predicted_tag:
                correct += 1

    accuracy = correct / total
    return accuracy
