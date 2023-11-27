import os
import warnings

warnings.filterwarnings("ignore")


class PathConfig:
    FILE_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.dirname(FILE_DIR)
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

    DATA_PATH = os.path.join(ROOT_DIR, "data")

    VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR, "vocab.txt")
    HMM_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "hmm.json")
    GREEDY_ALGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "greedy.json")
    VITERBI_ALGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "viterbi.json")

    ADVS_HMM_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "advs_hmm.json")
    ADVS_GREEDY_ALGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "advs_greedy.json")
    ADVS_VITERBI_ALGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "advs_viterbi.json")


class WSJDatasetConfig:
    cols = ["index", "sentences", "labels"]

    train_file_path = os.path.join(PathConfig.DATA_PATH, "train.json")
    dev_file_path = os.path.join(PathConfig.DATA_PATH, "dev.json")
    test_file_path = os.path.join(PathConfig.DATA_PATH, "test.json")


class VocabConfig:
    UNKNOWN_TOKEN = "<unk>"
    THRESHOLD = 2
    FILE_HEADER = ["word", "index", "frequency"]

    VOCAB_FILE = PathConfig.VOCAB_FILE_PATH


class HMMConfig:
    HMM_MODEL_SAVED = PathConfig.HMM_MODEL_SAVE_PATH
    ADVS_HMM_MODEL_SAVED = PathConfig.ADVS_HMM_MODEL_SAVE_PATH
