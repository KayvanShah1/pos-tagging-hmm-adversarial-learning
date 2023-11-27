import os

from decoder import GreedyDecoding, ViterbiDecoding, calculate_accuracy
from adv_hmm import HMM
from utils import (
    WallStreetJournalDataset,
    WSJDatasetConfig,
    VocabConfig,
    VocabularyGenerator,
    PathConfig,
)


def train_and_evaluate():
    # Prepare dataset
    print("\nReading and preparing data ...")
    train_dataset = WallStreetJournalDataset(path=WSJDatasetConfig.train_file_path)
    df_train = train_dataset.prepare_dataset()

    valid_dataset = WallStreetJournalDataset(path=WSJDatasetConfig.dev_file_path)
    df_valid = valid_dataset.prepare_dataset()

    unp_test_df = WallStreetJournalDataset(path=WSJDatasetConfig.test_file_path)._read_data()

    test_dataset = WallStreetJournalDataset(path=WSJDatasetConfig.test_file_path)
    test_dataset.prepare_dataset()

    # Generate vocabulary
    print("\nGenerating vocabulary ...")
    vocab_generator = VocabularyGenerator(
        threshold=VocabConfig.THRESHOLD, unknown_token=VocabConfig.UNKNOWN_TOKEN, save=True
    )
    vocab_df = vocab_generator.generate_vocabulary(df_train, "sentence")
    print("Selected threshold for unknown words: ", VocabConfig.THRESHOLD)
    print("Vocabulary size: ", vocab_df.shape[0])
    print(
        "Total occurrences of the special token <unk>: ",
        int(vocab_df[vocab_df["word"] == "<unk>"].frequency),
    )

    # Extract unique part-of-speech tags
    unique_pos_tags = df_train.labels.explode().unique()
    unique_pos_tags = unique_pos_tags.tolist()

    # Preprocess Data for training and evaluation
    train_sentences_with_pos_tags = train_dataset.get_sentences_with_pos_tags()
    valid_sentences_with_pos_tags = valid_dataset.get_sentences_with_pos_tags()
    test_sentences_with_pos_tags = test_dataset.get_sentences_with_pos_tags()

    # Initialize, train, and save the model
    print("\nTraining the HMM model ...")
    model = HMM(vocab_file=VocabConfig.VOCAB_FILE, labels=unique_pos_tags)
    model.fit(train_sentences_with_pos_tags)

    # Get all the parameters as probability matrices
    p, t, e = model.get_all_probability_matrices
    print("Number of Transition Parameters =", len(t.flatten()))
    print("Number of Emission Parameters =", len(e.flatten()))

    # Save the model
    model.save_model()

    print("\nValidating on dev data and producing inference results for test data ...")

    greedy_decoder = GreedyDecoding(p, t, e, model.states, model.vocab)

    # Apply Greedy Decoding on development data
    predicted_dev_tags = greedy_decoder.decode(valid_sentences_with_pos_tags)

    acc = calculate_accuracy(predicted_dev_tags, df_valid.labels.tolist())
    print("\nGreedy Decoding Accuracy: ", round(acc, 4))

    # Apply Greedy Decoding on Test data
    predicted_test_tags = greedy_decoder.decode(test_sentences_with_pos_tags)

    df_greedy_preds = unp_test_df.copy(deep=True)
    df_greedy_preds["labels"] = predicted_test_tags

    df_greedy_preds.to_json(PathConfig.GREEDY_ALGO_OUTPUT_PATH, orient="records", indent=4)
    print(
        "Saved Greedy Decoding predictions to"
        f" {os.path.relpath(PathConfig.GREEDY_ALGO_OUTPUT_PATH)}"
    )

    viterbi_decoder = ViterbiDecoding(p, t, e, model.states, model.vocab)

    # Apply Viterbi Decoding on development data
    predicted_dev_tags_viterbi = viterbi_decoder.decode(valid_sentences_with_pos_tags)

    acc_v = calculate_accuracy(predicted_dev_tags_viterbi, df_valid.labels.tolist())
    print("\nViterbi Decoding Accuracy: ", round(acc_v, 4))

    # Apply Greedy Decoding on Test data
    predicted_test_tags_v = viterbi_decoder.decode(test_sentences_with_pos_tags)

    df_viterbi_preds = unp_test_df.copy(deep=True)
    df_viterbi_preds["labels"] = predicted_test_tags_v

    df_viterbi_preds.to_json(PathConfig.VITERBI_ALGO_OUTPUT_PATH, orient="records", indent=4)
    print(
        "Saved Viterbi Decoding predictions to"
        f" {os.path.relpath(PathConfig.VITERBI_ALGO_OUTPUT_PATH)}"
    )


if __name__ == "__main__":
    train_and_evaluate()
