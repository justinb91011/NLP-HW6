#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###
        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""


        # Assert that there are no emissions from EOS or BOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, (
            "Your expected emission counts from EOS and BOS are not all zero, "
            "meaning you've accumulated them incorrectly!"
        )

        # Update emission probabilities (self.B)
        self.B_counts += λ  # add λ for smoothing
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize

        # Structural zeroes for BOS and EOS in emission probabilities
        self.B[self.eos_t, :] = 0  # EOS_TAG shouldn't emit any word
        self.B[self.bos_t, :] = 0  # BOS_TAG shouldn't emit any word

        # Assert that there are no transitions to BOS and no transitions from EOS
        assert self.A_counts[:, self.bos_t].any() == 0, (
            "Your expected transition counts to BOS are not all zero, "
            "meaning you've accumulated them incorrectly!"
        )
        assert self.A_counts[self.eos_t, :].any() == 0, (
            "Your expected transition counts from EOS are not all zero, "
            "meaning you've accumulated them incorrectly!"
        )

        # Update transition probabilities (self.A)
        self.A_counts += λ  # add λ for smoothing
        if self.unigram:
            # In unigram mode, make the transition matrix have identical rows
            self.A = self.A_counts[0] / self.A_counts[0].sum()
            self.A = self.A.repeat(self.k, 1)  # Repeat the single row across all rows
        else:
            # Normalize each row in bigram mode
            self.A = self.A_counts / self.A_counts.sum(dim=1, keepdim=True)

        # Enforce structural zeroes in the transition matrix
        self.A[:, self.bos_t] = 0  # BOS_TAG should not be a destination tag
        self.A[self.eos_t, :] = 0  # EOS_TAG should not transition to any tag
        # Update transition probabilities (self.A).  
        # Don't forget to respect the settings self.unigram and λ.
        # See the init_params() method for a discussion of self.A in the
        # unigram case.
        

    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      
      
        dev_loss = loss(self)   # evaluate the model at the start of training
        
        old_dev_loss: float = dev_loss     # loss from the last epoch
        step: int = 0   # total number of sentences the model has been trained on so far      
        while step < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Forward-backward algorithm.
        if not hasattr(self, 'A_counts'):
            self._zero_counts()  # Ensure A_counts and B_counts exist and are initialized to zero

        # Iterate through each word-tag pair in the sentence
        for i in range(len(isent) - 1):
            tag_i, word_i = isent[i]
            next_tag_i, _ = isent[i + 1]
            
            # Ensure none of the indices are None before proceeding
            if tag_i is None or word_i is None or next_tag_i is None:
                continue

            # Increment the transition count between consecutive tags, excluding transitions to BOS_TAG and from EOS_TAG
            if tag_i < self.k and next_tag_i < self.k and next_tag_i != self.bos_t and tag_i != self.eos_t:
                self.A_counts[tag_i, next_tag_i] += mult
            
            # Increment the emission count for the current tag emitting the current word
            if tag_i < self.k and word_i < self.V and tag_i != self.eos_t and tag_i != self.bos_t:
                # Only count emissions for non-EOS and non-BOS tags
                self.B_counts[tag_i, word_i] += mult

        # Handle the transition from BOS to the first tag and the transition to EOS
        first_tag, _ = isent[0]
        last_tag, _ = isent[-1]

        # Ensure first_tag and last_tag are not None before updating counts
        if self.bos_t is not None and first_tag is not None and first_tag < self.k:
            self.A_counts[self.bos_t, first_tag] += mult  # Transition from BOS to the first tag
        if last_tag is not None and self.eos_t is not None and last_tag < self.k:
            self.A_counts[last_tag, self.eos_t] += mult  # Transition from the last tag to EOS
        

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign 
        # directly to each alpha[j] in turn.
        # Number of words in the sentence (excluding BOS and EOS)

        n = len(isent) - 1
        alpha = [torch.full((self.k,), float('-inf')) for _ in range(n + 2)]
        alpha[0][self.bos_t] = 0.0  # log(1) = 0 at BOS

        log_A = torch.log(self.A + 1e-10)  # small constant to prevent log(0)
        log_B = torch.log(self.B + 1e-10)

        for j in range(1, n + 2):
            if j < n + 1:
                w_j = isent[j][0]
                emit_probs = log_B[:, w_j] if w_j < self.V else torch.zeros(self.k)
            else:
                emit_probs = torch.zeros(self.k)  # EOS, no emission

            for t in range(self.k):
                alpha_prev = alpha[j - 1] + log_A[:, t]  # Transition from all tags to t
                alpha[j][t] = torch.logsumexp(alpha_prev, dim=0) + emit_probs[t]

        log_Z = alpha[-1][self.eos_t]
        self.alpha = alpha  # for potential backward pass
        return log_Z
            # Note: once you have this working on the ice cream data, you may
            # have to modify this design slightly to avoid underflow on the
            # English tagging data. See section C in the reading handout.


    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the backward
        probability). 
        
        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts.  These depend on the alpha
        values and log Z, which were stored for us (in self) by the forward
        pass."""

        # Pre-allocate beta just as we pre-allocated alpha.
        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = self.eye[self.eos_t]  # vector that is one-hot at EOS_TAG
       
        raise NotImplementedError   # you fill this in!

        return log_Z_backward

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        isent = self._integerize_sentence(sentence, corpus)
        n = len(isent) - 1
        k = self.k

        # Initialize alpha_hat (max probabilities) and backpointers
        alpha = [torch.full((k,), float('-inf')) for _ in isent]
        backpointers = [torch.zeros(k, dtype=torch.long) for _ in isent]

        # Log probabilities to prevent underflow
        log_A = torch.log(self.A)
        log_B = torch.log(self.B)

         # Initialization at position 0 (BOS)
        alpha[0][self.bos_t] = 0.0
        # Viterbi algorithm
        for j in range(1, len(isent)):
            w_j = isent[j][0]  # Word index at position j
            # Compute emission probabilities
            if w_j < self.V:
                emit_prob = log_B[:, w_j]
            else:
                # For BOS_WORD and EOS_WORD, emission probability is 1 (log(1) = 0)
                emit_prob = torch.zeros(k)
            # Compute the scores matrix by adding alpha[j - 1] to each column of log_A
            # This uses broadcasting
            scores = alpha[j - 1].unsqueeze(1) + log_A
            # For each t_j, find the max score over t_prev and the corresponding backpointer
            alpha_j, backpointer_j = torch.max(scores, dim=0)

            # Add emission probabilities
            alpha[j] = alpha_j + emit_prob
            # Save backpointers
            backpointers[j] = backpointer_j
        # Backtracking
        tags = [0] * len(isent)
        tags[-1] = self.eos_t  # Start from EOS_TAG

        for j in range(len(isent) - 1, 0, -1):
            tags[j - 1] = backpointers[j][tags[j]]

        # Construct the tagged sentence
        tagged_sentence = Sentence()
        for j in range(len(sentence)):
            word = sentence[j][0]
            tag_index = tags[j]
            tag = self.tagset[tag_index]
            tagged_sentence.append((word, tag))

        return tagged_sentence

    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> HiddenMarkovModel:
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)\
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        if model.__class__ != cls:
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {model_path}.")

        logger.info(f"Loaded model from {model_path}")
        return model
