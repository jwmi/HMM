# HMM
Julia implementation of hidden Markov models

## Features
- Handles continuous or discrete observations
- Handles discrete hidden states only
- Customizable emission distributions
- Pre-implemented emission distribution models:
-- Discrete (on a finite set {1,...,K})
-- Univariate normal
-- Multivariate normal with diagonal covariance

## Algorithms implemented
- Viterbi algorithm
- Forward algorithm
- Backward algorithm
- Baum-Welch algorithm (expectation-maximization) for parameter estimation
- Generate data from an HMM

## Licensing

If you use this software in your research, please cite:<br>
Jeffrey W. Miller (2016). *Lecture Notes on Advanced Stochastic Modeling*. Duke University, Durham, NC.

Copyright (c) 2016 Jeffrey W. Miller. 
This software is released under the MIT License (see LICENSE).

