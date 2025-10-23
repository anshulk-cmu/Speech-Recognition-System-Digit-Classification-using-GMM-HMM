import numpy as np


class DiagGauss:
    def __init__(self, feature_size: int):
        """
        DiagGauss is the class of multivariate Gaussian distribution with diagonal covariance

        Args:
            feature_size (int): the number of feature (dimension) in Gaussian
        """
        self.feature_size = feature_size

        self.mean = np.zeros(self.feature_size)
        self.std = np.ones(self.feature_size)

        # const is the log of the constant independent of observation,
        # you should utilize this to speed up your logpdf
        self.const = 0.0

        # update std and const
        self.set_std(self.std)

    def set_mean(self, mean: np.ndarray) -> None:
        """
        update the mean of your guassian

        Args:
            mean (np.ndarray): np array with the shape of [D] where D is the feature size

        Returns:
        """

        assert len(mean.shape) == 1, 'mean should have one dimension'
        assert mean.shape[0] == self.feature_size, 'mean dim should match with the feature size'

        self.mean = mean

    def set_std(self, std: np.ndarray) -> None:
        """
        update the standard deviation of your guassian

        Args:
            std (np.ndarray): np array with the shape of [D] where D is the feature size

        Returns:
        """

        assert len(std.shape) == 1, 'std should have one dimension'
        assert std.shape[0] == self.feature_size, 'std dim should match with the feature size'

        self.std = std
        self.const = -0.5 * self.feature_size * np.log(2.0 * np.pi) - np.sum(np.log(self.std))


    def fit(self, X: np.ndarray) -> None:
        """
        fit the model, i.e., calculate mean and std

        Args:
            X (np.ndarray): X represents your sample matrix. It is two dimension numpy array with [N,D] shape
            where N is the sample size and D is the feature size

        Returns:
        """

        assert len(X.shape) == 2, 'X should have two dim'
        assert X.shape[1] == self.feature_size, 'the second dim should match with your feature size'

        # Implement your MLE
        # 1. Calculate the mean for each feature (column-wise average)
        mean = np.mean(X, axis=0)

        # 2. Calculate the variance (the average squared difference from the mean)
        var = np.mean((X - mean) ** 2, axis=0)

        # 3. Get the standard deviation (sqrt of variance)
        # We add a tiny value (1e-6) inside the sqrt to prevent errors
        # if the variance is zero (which can happen with unbalanced data)
        # This is a robust way to handle the tip in [cite: 123]
        std = np.sqrt(var + 1e-6)

        # 4. Store the values using set_mean and set_std
        self.set_mean(mean)
        self.set_std(std)


    def logpdf(self, X: np.ndarray):
        """
        compute the log pdf of your sample

        Args:
            X (np.ndarray): X represents your sample matrix. It is two dimension numpy array with [N,D] shape
            where N is the sample size and D is the feature size

        Returns:
            A 1 dimension numpy array of [N] shape where each element is the logpdf of X_i (i-th row of X)

        """

        assert len(X.shape) == 2, 'X should have two dim'
        assert X.shape[1] == self.feature_size, 'the second dim should match with your feature size'

        # We are implementing the log probability from Equation 4 
        # log(N) = const - 0.5 * sum( (o_d - u_d)^2 / r_d )
        #
        # The 'self.const' variable already holds the first two terms:
        # -D/2*log(2pi) - 0.5*sum(log(r_d)) 
        #
        # We just need to calculate the third term.

        # 1. Get the variance r from self.std (since r = std^2)
        variance = self.std ** 2

        # 2. Calculate (o - u)^2
        diff_sq = (X - self.mean) ** 2

        # 3. Calculate (o - u)^2 / r for all samples and dimensions
        # This has shape [N, D]
        term3_matrix = diff_sq / variance

        # 4. Sum across the dimensions (axis=1) to get shape [N]
        term3_sum = np.sum(term3_matrix, axis=1)

        # 5. Combine with the constant term
        # This is our final log probability for each sample
        log_prob = self.const - 0.5 * term3_sum

        return log_prob