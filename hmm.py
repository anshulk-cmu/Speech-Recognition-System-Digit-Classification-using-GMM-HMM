import numpy as np
from scipy.special import logsumexp

from gmm import DiagGMM


class HMM:
    def __init__(self, state_size: int, component_size: int, feature_size: int):
        """
        HMM is the acoustic model

        Args:
            state_size (int): the number of state in HMM
            component_size (int): the number of Gaussian in each GMM
            feature_size (int): the number of feature size in each Gaussian
        """

        self.state_size = state_size
        self.component_size = component_size
        self.feature_size = feature_size

        # For each state in HMM, we would like to remember all frames that aligned to them
        self.state2frames = [[] for i in range(self.state_size)]

        # Similarly, for each state, we would like to remember its recursion and forward occurance in all alignments.
        self.state2count = [[0, 0] for i in range(self.state_size)]

        # initialize transition probability
        self.trans_logpdf = np.zeros((self.state_size, self.state_size))

        # create your GMM
        self.gmms = []
        for i in range(self.state_size):
            self.gmms.append(DiagGMM(self.component_size, self.feature_size))

    def initialize(self):
        """
        Initialize your HMM and GMM

        Returns:
        """

        # initialize transition model
        self.trans_logpdf = np.zeros((self.state_size, self.state_size))

        # we follow Kaldi's style by setting recursive/forward prob to 0.75/0.25.
        # Feel free to change it to other parameters if that gives you better performance
        for i in range(self.state_size-1):
            self.trans_logpdf[i][i] = np.log(0.75)
            self.trans_logpdf[i][i+1] = np.log(0.25)

        # initialize emission model
        for i in range(self.state_size):
            frame_lst = self.state2frames[i]
            frames = np.stack(frame_lst)
            self.gmms[i].initialize(frames)

        self.clean_accumulation()

    def clean_accumulation(self):
        """
        After one EM step, you might want to clean your existing alignments.

        Returns:

        """
        self.state2frames = [[] for i in range(self.state_size)]
        self.state2count = [[0, 0] for i in range(self.state_size)]

    def align_equally(self, X: np.ndarray):
        """
        Before training the GMM-HMM model, we would like to obtain a good initialization parameters especially for the GMM.
        We assume that each state is all equally aligned with the observation.
        For example if the number of frames in X is 10 and there are 5 states,
        we might want to align X with [0,0,1,1,2,2,3,3,4,4]

        Args:
            X (np.ndarray): a single sample point of [N,D] where $N$ is the number of frames and $D$ is the feature size

        Returns:
            alignments (np.ndarray): one dimension numpy array with shape [N], each element should be an int between [0, state_size-1]

        """

        assert len(X.shape) == 2, "sample should have two dimension"
        assert X.shape[1] == self.feature_size, "feature size does not match"

        frame_size = X.shape[0]
        frame_per_state = int(frame_size / self.state_size)

        alignment = np.ones(frame_size, dtype=np.int32)*(self.state_size-1)

        for i in range(self.state_size):
            alignment[i*frame_per_state:(i+1)*frame_per_state] = i

        return alignment

    def align(self, X: np.ndarray) -> np.ndarray:
        """
        This is the E step of HMM, in which we would like to align each frame in X to a HMM state.
        Implement the Viterbi search described in the handout to find the good alignment.
        You might want to do most of the computation in the log space by using logpdf in GMM and trans_logprob

        Args:
            X (np.ndarray): a single sample point of [N,D] where $N$ is the number of frames and $D$ is the feature size

        Returns:
            alignments (np.ndarray): one dimension numpy array with shape [N], each element should be an int between [0, state_size-1]
            You can check the returned value of align_equally as an example.
        """

        time_size = X.shape[0]
        feature_size = X.shape[1]

        assert self.feature_size == feature_size

        # compute all probs at once
        emission_pdf = np.zeros((self.state_size, time_size))
        for i, gmm in enumerate(self.gmms):
            emission_pdf[i] = gmm.logpdf(X)

        # viterbi search graph
        graph = -np.inf * np.ones((self.state_size, time_size))

        # remember max previous pointer
        prev_pointer = np.zeros((self.state_size, time_size), dtype=np.int32)

        # start point is (0,0)
        # Prob of being in state 0 at time 0 is just the emission prob
        graph[0, 0] = emission_pdf[0, 0]

        # implement your viterbi search
        # Loop through all time steps (frames)
        for t in range(1, time_size):
            # Loop through all states
            for j in range(self.state_size):
                # We are implementing Equation 15
                
                # 1. Prob of staying in the same state (j -> j)
                prob_stay = graph[j, t-1] + self.trans_logpdf[j, j]
                
                # 2. Prob of moving from the previous state (j-1 -> j)
                # (Only possible if j > 0)
                prob_move = -np.inf
                if j > 0:
                    prob_move = graph[j-1, t-1] + self.trans_logpdf[j-1, j]

                # 3. Find the max of the two paths
                best_prev_prob = np.maximum(prob_stay, prob_move)

                # 4. Store the path pointer
                if prob_stay > prob_move:
                    prev_pointer[j, t] = j
                else:
                    prev_pointer[j, t] = j - 1
                
                # 5. Add the emission probability for the current cell
                graph[j, t] = emission_pdf[j, t] + best_prev_prob

        # --- Backtracking ---
        # Find the best path by starting from the end
        alignment = np.zeros(time_size, dtype=np.int32)
        
        # Start at the last state at the last time step
        current_state = self.state_size - 1
        alignment[time_size - 1] = current_state

        # Follow the pointers backward
        for t in range(time_size - 2, -1, -1):
            current_state = prev_pointer[current_state, t + 1]
            alignment[t] = current_state
        
        return alignment

    def accumulate(self, X: np.ndarray, alignments: np.ndarray) -> None:
        """
        Accumulate the aligned frames/counts for each state. Those statistics would be latter used to update GMM/HMM

        Args:
            X (np.ndarray): sample matrix, it is a [N, H] shape matrix where N is the frame size, H is the feature size
            alignments (np.ndarray): alignment returned from align or align_equally. it is a [N] shaped numpy array.

        Returns:
        """

        prev_id = -1

        for i, state_id in enumerate(alignments):

            # accumulate stats for emission
            self.state2frames[state_id].append(X[i])

            # accumulate stats for transition
            if prev_id >= 0:
                if prev_id == state_id:

                    # inc transition from prev_id to itself
                    self.state2count[prev_id][0] += 1
                else:
                    # inc transition from prev_id to next id
                    self.state2count[prev_id][1] += 1

            prev_id = state_id

    def update(self):
        """
        Use existing accumulated alignments to update GMM and HMM.
        Do not forget to clean your accumulation.

        Returns:
        """

        # 1. Update Transition Probabilities (HMM)
        # Based on Equations 13 and 14 [cite: 201, 208]
        
        # Re-initialize to -inf
        new_trans_logpdf = -np.inf * np.ones((self.state_size, self.state_size))

        for i in range(self.state_size):
            recursive_count = self.state2count[i][0]
            forward_count = self.state2count[i][1]
            total_count = recursive_count + forward_count
            
            if total_count > 0:
                # Calculate new prob based on counts
                prob_stay = recursive_count / total_count
                prob_move = forward_count / total_count
            else:
                # No data for this state, use default (from initialize)
                prob_stay = 0.75
                prob_move = 0.25

            new_trans_logpdf[i, i] = np.log(prob_stay + 1e-6) # Add 1e-6 for stability
            if i + 1 < self.state_size:
                new_trans_logpdf[i, i+1] = np.log(prob_move + 1e-6)
        
        self.trans_logpdf = new_trans_logpdf

        # 2. Update Emission Probabilities (GMMs) [cite: 270]
        # We re-train each GMM using the frames assigned to it
        for i in range(self.state_size):
            frame_lst = self.state2frames[i]
            
            # Only update if we have frames for this state
            if len(frame_lst) > 0:
                frames = np.stack(frame_lst)
                # Call the gmm.fit() you implemented earlier!
                self.gmms[i].fit(frames)

        # 3. Clean up for the next EM iteration
        self.clean_accumulation()

    def logpdf(self, X):
        """
        compute the marginalized log probability of X

        Args:
            X (): sample matrix, it is a [N, H] shape matrix where N is the frame size, H is the feature size

        Returns: log probability
        """

        time_size = X.shape[0]
        feature_size = X.shape[1]

        assert self.feature_size == feature_size

        # compute all probs at once
        emission_pdf = np.zeros((self.state_size, time_size))
        for i, gmm in enumerate(self.gmms):
            emission_pdf[i] = gmm.logpdf(X)

        # viterbi search graph
        graph = -np.inf*np.ones((self.state_size, time_size))

        # start point is (0,0)
        graph[0, 0] = emission_pdf[0, 0]

        # forward path (almost similar to the viterbi search)
        # This implements Equation 16 
        for t in range(1, time_size):
            for j in range(self.state_size):
                
                # 1. Prob of staying in the same state (j -> j)
                prob_stay = graph[j, t-1] + self.trans_logpdf[j, j]
                
                # 2. Prob of moving from the previous state (j-1 -> j)
                prob_move = -np.inf
                if j > 0:
                    prob_move = graph[j-1, t-1] + self.trans_logpdf[j-1, j]
                
                # 3. Sum the probabilities (in log space)
                # This is the ONLY difference from Viterbi's align()
                sum_prev_probs = logsumexp([prob_stay, prob_move])

                # 4. Add the emission probability for the current cell
                graph[j, t] = emission_pdf[j, t] + sum_prev_probs
        
        # The final log probability is the total prob of ending
        # in the last state at the last time step [cite: 280]
        return graph[self.state_size - 1, time_size - 1]
