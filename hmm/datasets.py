import os

import pandas as pd

from utils import VocabConfig, WSJDatasetConfig


class WallStreetJournalDataset:
    def __init__(self, path, split="train"):
        self.path = path
        self.split = split

        self.data: pd.DataFrame = None
        self.cols = WSJDatasetConfig.cols

    def _read_data(self):
        self.data = pd.read_json(self.path)
        return self.data

    def _process_sentences(self):
        self.data["sentence"] = self.data["sentence"].apply(
            lambda sentence: [word.lower() for word in sentence],
        )

    def prepare_dataset(self):
        self._read_data()
        self._process_sentences()
        return self.data

    def get_sentences_with_pos_tags(self):
        if "labels" in self.data.columns:
            sentences_with_pos_tags = self.data.loc[:, ["sentence", "labels"]].apply(
                lambda row: list(zip(row["sentence"], row["labels"])), axis=1
            )
        else:
            sentences_with_pos_tags = self.data["sentence"].apply(
                lambda sentence: list(zip(sentence, [None] * len(sentence)))
            )
        sentences_with_pos_tags = sentences_with_pos_tags.tolist()
        return sentences_with_pos_tags


class VocabularyGenerator:
    def __init__(
        self, threshold: int, unknown_token: str = None, save: bool = False, path: str = None
    ):
        """Initialize a VocabularyGenerator

        Args:
            threshold (int): Frequency threshold for rare words.
            unknown_token (str, optional): Token to replace rare words. Defaults to None.
            save (bool, optional): Flag to save the vocabulary. Default is True.
            path (str, optional): Path to save the vocabulary. Defaults to None.

        Usage:
            vocab_generator = VocabularyGenerator(threshold=3, unknown_token="<unk>")
            vocab_df = vocab_generator.generate_vocabulary(data, "sentence")
        """
        self.threshold = threshold
        self.unknown_token = (
            unknown_token if unknown_token is not None else VocabConfig.UNKNOWN_TOKEN
        )
        self._save = save

        if self._save and path is None:
            self.path = VocabConfig.VOCAB_FILE
        else:
            self.path = path

    def _count_word_frequency(self, data, sentence_col_name):
        word_freq = (
            data[sentence_col_name]
            .explode()
            .value_counts()
            .rename_axis("word")
            .reset_index(name="frequency")
        )
        return word_freq

    def generate_vocabulary(self, data: pd.DataFrame, sentence_col_name: str):
        """Generate a vocabulary from the provided dataset.

        Args:
            data (pd.DataFrame): The DataFrame containing the dataset.
            sentence_col_name (str): The name of the column containing sentences.

        Returns:
            pd.DataFrame: A DataFrame with the generated vocabulary.

        This method takes a DataFrame with sentences and generates a vocabulary based on word
        frequencies. It replaces words with frequencies less than the specified threshold with
        the unknown token ("<unk>"). The resulting DataFrame is sorted by frequency and indexed.

        If the 'save' flag is set, the vocabulary will be saved to the specified path.

        Usage:
            ```py
            vocab_generator = VocabularyGenerator(threshold=3, unknown_token="<unk>")
            vocab_df = vocab_generator.generate_vocabulary(data, sentence_col_name)
            ```
        """
        word_freq_df = self._count_word_frequency(data, sentence_col_name)

        # Replace words with frequency less than threshold with '<unk>'
        word_freq_df["word"] = word_freq_df.apply(
            lambda row: self.unknown_token if row["frequency"] <= self.threshold else row["word"],
            axis=1,
        )

        # Group by 'Word' and aggregate by sum
        word_freq_df = word_freq_df.groupby("word", as_index=False)["frequency"].agg("sum")

        # Sort the DataFrame by frequency
        word_freq_df = word_freq_df.sort_values(by="frequency", ascending=False, ignore_index=True)

        # Placing Special Tokens at the top of the DataFrame
        unk_df = word_freq_df.loc[word_freq_df["word"] == self.unknown_token]
        word_freq_df = word_freq_df.loc[word_freq_df["word"] != self.unknown_token]

        word_freq_df = pd.concat([unk_df, word_freq_df], ignore_index=True)

        # Add an index column
        word_freq_df["index"] = range(len(word_freq_df))

        if self._save:
            self.save_vocab(word_freq_df, self.path)

        return word_freq_df

    def save_vocab(self, word_freq_df, path):
        """Write your vocabulary to the file"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "w") as file:
            vocabulary = word_freq_df.to_records(index=False)
            for word, frequency, index in vocabulary:
                file.write(f"{word}\t{index}\t{frequency}\n")

        print(f"Saved vocabulary to file {os.path.relpath(path)}")
