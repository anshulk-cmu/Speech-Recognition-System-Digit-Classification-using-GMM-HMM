import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from gauss import DiagGauss


class DiagGMM:
    def __init__(self, component_size: int, feature_size: int):
        """
        DiagGMM is the class of Gaussian Mixture Model whose components are diagonal Gaussian distributions

        Args:
            component_size (int): the number of Gaussian mixtures
            feature_size (int): the number of feature in the diagonal Gaussian distribution
        """
        self.component_size = component_size
        self.feature_size = feature_size

        # mixture weight
        self.log_weight = None

        # all gaussian distributions
        self.gauss = None


    def initialize(self, X: np.ndarray):
        """
        Initialize the GMM model with sample X

        Args:
            X (np.ndarray): the sample matrix, same as fit interface

        Returns:

        """

        # initialize all gauss distribution
        self.gauss = [DiagGauss(self.feature_size) for i in range(self.component_size)]

        # initial weight
        self.log_weight = np.log(np.ones(self.component_size)/self.component_size)

        # you can use kmeans to do the initialize the model
        kmeans_cluster = KMeans(n_clusters=self.component_size)

        sample_size = X.shape[0]

        # initialize with hard assignments
        assignment = kmeans_cluster.fit_predict(X)

        # update each component via equal alignment
        for i in range(self.component_size):
            self.log_weight[i] = np.log(np.sum(assignment == i)/sample_size)

            # extract all frame assigned to i
            X_i = X[assignment == i]

            # compute MLE mean and std
            mean = np.mean(X_i, axis=0)
            std = np.sqrt(np.sum(((X_i - mean)**2.0), axis=0)/len(X_i))

            # update mean and std
            self.gauss[i].set_mean(mean)
            self.gauss[i].set_std(std)


    def E_step(self, X: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute the latent responsibilies for each sample and component
        Args:
            X (np.ndarray): [N, D] matrix where N is the sample size and D is the feature size

        Returns:
            a matrix with the size [N, C] where C is the component size and N is sample size.
            The ij-entry is the responsibility of j-th component in i-th sample

        """

        # We are implementing Equation 9
        # This is (weight_k * prob(X | k)) / sum_k'(weight_k' * prob(X | k'))
        # To avoid numbers getting too small (underflow), we do this in log-space.

        # 1. Get the log-likelihood of X for each component k
        # log_prob_matrix will have shape [N, C]
        log_prob_matrix = np.zeros((X.shape[0], self.component_size))
        for k in range(self.component_size):
            # log(prob(X | k)) + log(weight_k)
            log_prob_matrix[:, k] = self.gauss[k].logpdf(X) + self.log_weight[k]

        # 2. Get the log-denominator (the log of the total probability for each sample)
        # logsumexp sums probabilities in the log domain 
        # log_denominator shape is [N]
        log_denominator = logsumexp(log_prob_matrix, axis=1, keepdims=True)

        # 3. Calculate log-responsibility: log(numerator) - log(denominator)
        log_responsibilities = log_prob_matrix - log_denominator

        # 4. Convert back from log-space to probability space
        # The M-step equations expect probabilities, not log-probabilities [cite: 115]
        responsibilities = np.exp(log_responsibilities)

        return responsibilities


    def M_step(self, X, comp_weight) -> None:
        """
        Maximization step: use the responsibilies (comp_weight) to update your GMM model.
        In particular, you might want to update three parameters:
        - mixture weight (log_weight)
        - mean of each Gaussian component
        - std of each Gaussian component

        Args:
            X (np.ndarray): [N, D] matrix where N is the sample size and D is the feature size
            comp_weight (np.ndarray): [N, C] matrix of component responsibilities. It is the returned value from E step.
            C is the component size and N is sample size. The ij-cell is the responsibility of j-th component in i-th sample

        Returns:
        """

        # comp_weight is our gamma(k,t), shape [N, C]
        # X is our o_t, shape [N, D]

        # 1. Calculate sum of responsibilities for each component k
        # This is the denominator for all three update equations
        # gamma_k will have shape [C]
        gamma_k = np.sum(comp_weight, axis=0)
        
        # Add a small epsilon for numerical stability
        gamma_k_stable = gamma_k + 1e-6

        # 2. Update mixture weights (Equation 10) 
        gamma_total = np.sum(gamma_k)
        self.log_weight = np.log(gamma_k / (gamma_total + 1e-6))

        # 3. Update means (Equation 11) [cite: 102]
        # Numerator is sum_t(gamma_k(t) * o_t).
        # We can compute this for all k and d at once with a dot product.
        mean_numerator = comp_weight.T @ X  # Shape [C, D]
        new_mean = mean_numerator / gamma_k_stable[:, np.newaxis]

        # 4. Update variances (Equation 12) [cite: 103]
        # Numerator is sum_t(gamma_k(t) * (o_t - new_mean_k)^2)
        # We vectorize this:
        # (X[n,d] - new_mean[k,d])^2 -> shape [N, C, D]
        diff_sq = (X[:, np.newaxis, :] - new_mean[np.newaxis, :, :]) ** 2
        # comp_weight[n,k] * diff_sq[n,k,d] -> shape [N, C, D]
        weighted_diff_sq = comp_weight[:, :, np.newaxis] * diff_sq
        # sum over n -> shape [C, D]
        var_numerator = np.sum(weighted_diff_sq, axis=0)

        new_var = var_numerator / gamma_k_stable[:, np.newaxis]

        # 5. Get new std and update all Gaussian components
        new_std = np.sqrt(new_var + 1e-6)
        
        # Apply threshold as suggested in Tip 5 
        new_std = np.maximum(new_std, 1e-3)

        for k in range(self.component_size):
            self.gauss[k].set_mean(new_mean[k, :])
            self.gauss[k].set_std(new_std[k, :])


    def fit(self, X: np.ndarray):
        """
        fit the GMM model with your sample X.
        You should update your model iteratively with EM algorithm

        Args:
            X (np.ndarray): sample matrix of shape [N, D] where N is the number of sample (frame),
            D is the feature size

        Returns:
        """

        # estimate the GMM with kmeans
        if self.gauss is None:
            self.initialize(X)

        # EM steps
        for i in range(40):

            # compute the responsibility
            comp_weight = self.E_step(X)

            # compute
            self.M_step(X, comp_weight)


    def logpdf(self, X: np.ndarray):
        """
        compute the GMM logpdf of a sample

        Args:
            X (np.ndarray): sample matrix of shape [N, D] where N is the number of sample (frame)
            D is the feature size

        Returns:
            an np array of shape [N] where each element is the logpdf of X_i (the i-th row in X)
        """

        logprob_lst = []
        for i in range(self.component_size):
            logprob_lst.append(self.gauss[i].logpdf(X) + self.log_weight[i])

        # sum probability
        logprob = logsumexp(logprob_lst, axis=0)
        return logprob