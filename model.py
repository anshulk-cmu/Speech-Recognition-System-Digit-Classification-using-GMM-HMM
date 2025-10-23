from typing import List
import numpy as np
from tqdm import tqdm
import sklearn.metrics

from hmm import HMM


class DigitClassification:
    def __init__(self, state_size: int, component_size: int, feature_size: int, vocab_size: int, num_epoch: int):
        """
        Digit Classification Model

        Args:
            state_size (int): number of HMM state per word
            component_size (int): number of gaussian per GMM
            feature_size (int): number of feature in gaussian
            vocab_size (int): number of classes to predict (digits in our case)
            num_epoch (int): number of epochs
        """

        self.state_size = state_size
        self.component_size = component_size
        self.feature_size = feature_size
        self.vocab_size = vocab_size
        self.num_epoch = num_epoch

        self.hmms = []

        for i in range(vocab_size):
            self.hmms.append(HMM(state_size, component_size, feature_size))

    def initialize(self, X_train:List[np.ndarray], y_train: List[int]) -> None:
        """
        Initialize HMM and GMM with equal alignments

        Args:
            X_train (List[np.ndarray]): a list of samples, each sample has the shape [N, D] where N is the frame size and D is the feature size
            y_train (List[int]): a list of labels, each element is in [0, vocab_size-1]
        """

        # process each sample
        for i, y in enumerate(y_train):

            # get equal alignments from HMM
            alignment = self.hmms[y].align_equally(X_train[i])

            # accumulate those alignments to HMM states
            self.hmms[y].accumulate(X_train[i], alignment)

        # initialize all of HMMs
        for hmm in self.hmms:
            hmm.initialize()


    def fit(self, X_train:List[np.ndarray], y_train: List[int]) -> None:
        """
        Train HMM and GMM

        Args:
            X_train (List[np.ndarray]): a list of samples, each sample has the shape [N, D] where N is the frame size and D is the feature size
            y_train (List[int]): a list of labels, each element is in [0, vocab_size-1]
        """

        # initialize your HMM and GMM
        self.initialize(X_train, y_train)
        print("initial accuracy: ", self.validate(X_train, y_train))

        # training loop
        for ii in range(self.num_epoch):

            for i, y in tqdm(enumerate(y_train), total=len(y_train), desc=f"Fit (epoch {ii}):"):
                X = X_train[i]

                # use current HMM to extract alignment
                alignment = self.hmms[y].align(X)

                # accumulate those alignment to states
                self.hmms[y].accumulate(X, alignment)

            for hmm in self.hmms:
                hmm.update()

            print("Train set acc: ", self.validate(X_train, y_train))


    def predict(self, X_test: List[np.ndarray]) -> List[int]:
        """
        find the most likely digit in the vocabulary

        Args:
            X_test (List[np.ndarray):  a list of samples, each sample has the shape [N, D] where N is the frame size and D is the feature size

        Returns:
            a list of labels, each label should be an integer between [0, vocab_size-1]
        """

        result = []

        for X in tqdm(X_test, total=len(X_test), desc="Predict"):
            max_label = -1
            max_prob = -np.inf

            for i, hmm in enumerate(self.hmms):
                prob = hmm.logpdf(X)
                if prob > max_prob:
                    max_label = i
                    max_prob = prob

            result.append(max_label)

        return result


    def validate(self, X_dev:List[np.ndarray], y_dev: List[int]) -> float:
        """
        validate your model

        Args:
            X_dev (List[np.ndarray]): a list of samples, each sample has the shape [N, D] where N is the frame size and D is the feature size
            y_dev (List[int]): a list of labels, each element is in [0, vocab_size-1]

        Returns:
            accuracy score: float
        """

        y_pred = self.predict(X_dev)

        return sklearn.metrics.accuracy_score(y_dev, y_pred)
