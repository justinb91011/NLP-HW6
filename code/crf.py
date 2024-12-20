#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Conditional Random Fields.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
from typing import Callable, Optional
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

import itertools, more_itertools
from tqdm import tqdm # type: ignore

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer
from hmm import HiddenMarkovModel

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomField(HiddenMarkovModel):
    """An implementation of a CRF that has only transition and 
    emission features, just like an HMM."""
    
    # CRF inherits forward-backward and Viterbi methods from the HMM parent class,
    # along with some utility methods.  It overrides and adds other methods.
    # 
    # Really CRF and HMM should inherit from a common parent class, TaggingModel.  
    # We eliminated that to make the assignment easier to navigate.
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize params self.WA and self.WB to small random values, and
        then compute the potential matrices A, B from them.
        As in the parent method, we respect structural zeroes ("Don't guess when you know")."""

        k = self.k
        V = self.V

        # For a unigram model, self.WA should just have a single row
        rows = 1 if self.unigram else k

        # Initialize self.WA and self.WB to small random values
        self.WA = 0.01 * torch.randn(rows, k)
        self.WB = 0.01 * torch.randn(k, V)

        
        NEG_INF = 1e-10
        self.WA[:, self.bos_t] = NEG_INF
        self.WB[self.bos_t, :] = NEG_INF
        self.WB[self.eos_t, :] = NEG_INF
        self.updateAB()   # compute potential matrices

    def updateAB(self) -> None:
        """Set the transition and emission matrices self.A and self.B, 
        based on the current parameters self.WA and self.WB.
        See the "Parametrization" section of the reading handout."""
       
        if self.unigram:
            # For a unigram model, expand WA to a full matrix
            A_potentials = torch.exp(self.WA).repeat(self.k, 1)
        else:
            A_potentials = torch.exp(self.WA)

        B_potentials = torch.exp(self.WB)
        self.A = A_potentials  # Transition potentials
        self.B = B_potentials  # Emission potentials

        # Compute log-potentials
        self.log_A = self.WA.repeat(self.k, 1) if self.unigram else self.WA
        self.log_B = self.WB

    @override
    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[ConditionalRandomField], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              eval_interval: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              max_steps: int = 50000,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the CRF on the given training corpus, starting at the current parameters.

        The minibatch_size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus,
        which yields batch gradient ascent instead of stochastic gradient ascent.)
        
        The eval_interval controls how often we evaluate the loss function (which typically
        evaluates on a development corpus).
        
        lr is the learning rate, and reg is an L2 batch regularization coefficient.

        We always do at least one full epoch so that we train on all of the sentences.
        After that, we'll stop after reaching max_steps, or when the relative improvement 
        of the evaluation loss, since the last evalbatch, is less than the
        tolerance.  In particular, we will stop when the improvement is
        negative, i.e., the evaluation loss is getting worse (overfitting)."""
        
        def _loss() -> float:
            # Evaluate the loss on the current parameters.
            # This will print its own log messages.
            # 
            # In the next homework we will extend the codebase to use backprop, 
            # which finds gradient with respect to the parameters.
            # However, during evaluation on held-out data, we don't need this
            # gradient and we can save time by turning off the extra bookkeeping
            # needed to compute it.
            with torch.no_grad():  # type: ignore 
                return loss(self)      

        # This is relatively generic training code.  Notice that the
        # updateAB() step before each minibatch produces A, B matrices
        # that are then shared by all sentences in the minibatch.
        #
        # All of the sentences in a minibatch could be treated in
        # parallel, since they use the same parameters.  The code
        # below treats them in series -- but if you were using a GPU,
        # you could get speedups by writing the forward algorithm
        # using higher-dimensional tensor operations that update
        # alpha[j-1] to alpha[j] for all the sentences in the
        # minibatch at once.  PyTorch could then take better advantage
        # of hardware parallelism on the GPU.

        if reg < 0: raise ValueError(f"{reg=} but should be >= 0")
        if minibatch_size <= 0: raise ValueError(f"{minibatch_size=} but should be > 0")
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        min_steps = len(corpus)   # always do at least one epoch
        self._zero_grad()     # get ready to accumulate their gradient
        steps = 0
        old_loss = _loss()    # evaluate initial loss
        for evalbatch in more_itertools.batched(
                           itertools.islice(corpus.draw_sentences_forever(), 
                                            max_steps),  # limit infinite iterator
                           eval_interval): # group into "evaluation batches"
            for sentence in tqdm(evalbatch, total=eval_interval):
                # Accumulate the gradient of log p(tags | words) on this sentence 
                # into A_counts and B_counts.
                self.accumulate_logprob_gradient(sentence, corpus)
                steps += 1
                
                if steps % minibatch_size == 0:              
                    # Time to update params based on the accumulated 
                    # minibatch gradient and regularizer.
                    self.logprob_gradient_step(lr)
                    self.reg_gradient_step(lr, reg, minibatch_size / len(corpus))
                    self.updateAB()      # update A and B potential matrices from new params
                    self._zero_grad()    # get ready to accumulate a new gradient for next minibatch
            
            # Evaluate our progress.
            curr_loss = _loss()
            if steps >= min_steps and curr_loss >= old_loss * (1-tolerance):
                break   # we haven't gotten much better since last evalbatch, so stop
            old_loss = curr_loss   # remember for next evalbatch

        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
 
    @override
    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Return the *conditional* log-probability log p(tags | words) under the current
        model parameters.  This behaves differently from the parent class, which returns
        log p(tags, words).
        
        Just as for the parent class, if the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  Note that if the sentence is completely
        untagged, then the marginal probability will be 1.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)

        # Remove all tags and re-integerize the sentence.
        # Working with this desupervised version will let you sum over all taggings
        # in order to compute the normalizing constant for this sentence.
        desup_isent = self._integerize_sentence(sentence.desupervise(), corpus)

        # Compute log numerator (sum over paths consistent with the observed tags)
        log_numerator = self.forward_pass(isent)

        # Compute log Z(w), the normalization constant
        log_Z = self.forward_pass(desup_isent)

        # Compute the conditional log-probability
        log_prob = log_numerator - log_Z

        return log_prob

    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Add the gradient of self.logprob(sentence, corpus) into a total minibatch
        gradient that will eventually be used to take a gradient step."""
        
        # In the present class, the parameters are self.WA, self.WB, the gradient
        # is a difference of observed and expected counts, and you'll accumulate
        # the gradient information into self.A_counts and self.B_counts.  
        # 
        # (In the next homework, you'll have fancier parameters and a fancier gradient,
        # so you'll override this and accumulate the gradient using PyTorch's
        # backprop instead.)
        
        isent_sup   = self._integerize_sentence(sentence, corpus)
        isent_desup = self._integerize_sentence(sentence.desupervise(), corpus)

        # Zero counts for this sentence
        self._zero_counts()

        # Compute observed counts from the supervised sentence
        self._compute_observed_counts(isent_sup)
        observed_A_counts = self.A_counts.clone()
        observed_B_counts = self.B_counts.clone()

        # Compute expected counts from the model
        self._zero_counts()  # Reset counts before E_step
        self.E_step(isent_desup)
        expected_A_counts = self.A_counts.clone()
        expected_B_counts = self.B_counts.clone()

        # Compute gradients: observed counts - expected counts
        grad_A_counts = observed_A_counts - expected_A_counts
        grad_B_counts = observed_B_counts - expected_B_counts

        # Initialize gradients if they don't exist
        if not hasattr(self, 'grad_WA'):
            self.grad_WA = torch.zeros_like(self.WA)
            self.grad_WB = torch.zeros_like(self.WB)

        # For unigram, sum over rows
        if self.unigram:
            grad_WA = grad_A_counts.sum(dim=0).unsqueeze(0)
        else:
            grad_WA = grad_A_counts

        # Accumulate gradients for the current sentence
        self.grad_WA += grad_WA
        self.grad_WB += grad_B_counts


        
    def _zero_grad(self):
        """Reset the gradient accumulator to zero."""
        # You'll have to override this method in the next homework; 
        # see comments in accumulate_logprob_gradient().
        self._zero_counts()
    
        # Reset gradients
        if hasattr(self, 'grad_WA'):
            self.grad_WA.zero_()
            self.grad_WB.zero_()
        else:
            # Initialize gradients if they don't exist
            self.grad_WA = torch.zeros_like(self.WA)
            self.grad_WB = torch.zeros_like(self.WB)

    def logprob_gradient_step(self, lr: float) -> None:
        """Update the parameters using the accumulated logprob gradient.
        lr is the learning rate (stepsize)."""
        
        if self.unigram:
            # self.WA is [1,k], grad_WA is [1,k]
            self.WA += lr * self.grad_WA
        else:
            self.WA += lr * self.grad_WA
        
        self.WB += lr * self.grad_WB
        
    def reg_gradient_step(self, lr: float, reg: float, frac: float):
        """Update the parameters using the gradient of our regularizer.
        More precisely, this is the gradient of the portion of the regularizer 
        that is associated with a specific minibatch, and frac is the fraction
        of the corpus that fell into this minibatch."""
                    
        # Because this brings the weights closer to 0, it is sometimes called
        # "weight decay".
        
        if reg == 0: return      # can skip this step if we're not regularizing

        # Decay factor
        decay = 1 - 2 * lr * reg * frac
        self.WA *= decay
        self.WB *= decay

    def _compute_observed_counts(self, isent: IntegerizedSentence) -> None:
        """Compute the observed counts from the supervised sentence and accumulate them into self.A_counts and self.B_counts."""
        n = len(isent)
        for j in range(1, n):
            t_prev = isent[j - 1][1]
            t_curr = isent[j][1]
            w_curr = isent[j][0]

            if t_prev is None or t_curr is None:
                continue  # Skip if tags are missing

            # Ensure indices are within bounds
            if t_prev >= self.k or t_curr >= self.k:
                continue  # Skip invalid tag indices

            # Transition counts
            if self.unigram:
                self.A_counts[0, t_curr] += 1
            else:
                self.A_counts[t_prev, t_curr] += 1

            # Emission counts
            if w_curr < self.V:
                self.B_counts[t_curr, w_curr] += 1
            else:
                # w_curr corresponds to BOS_WORD or EOS_WORD, which are not included in self.B_counts
                pass  # No emission count is added for BOS_WORD or EOS_WORD
    