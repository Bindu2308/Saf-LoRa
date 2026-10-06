"""
receiver/hmm/hmm_config.py

Parameters for the 2-state (GOOD/BAD) interference HMM. These are
reasonable starting defaults, not learned from data yet -- tune them once
you have enough real logged SNR sequences to fit better transition/emission
parameters (a Baum-Welch refit is a natural next step, out of scope for
this first working version).
"""

# State transition probabilities. High diagonal = interference state
# tends to persist for a while rather than flicker every observation,
# matching how real RF interference (e.g. a nearby transmitter, weather)
# tends to have some time correlation.
P_GOOD_TO_GOOD = 0.90
P_GOOD_TO_BAD = 1 - P_GOOD_TO_GOOD
P_BAD_TO_BAD = 0.70
P_BAD_TO_GOOD = 1 - P_BAD_TO_BAD

# Emission model: SNR (dB) is modeled as Gaussian conditioned on state.
# These starting values assume your typical GOOD-state SNR (from working
# links) is around 12-13 dB (matches observed real logs), with BAD state
# being a meaningfully worse, more variable regime.
GOOD_SNR_MEAN_DB = 12.5
GOOD_SNR_STD_DB = 1.5
BAD_SNR_MEAN_DB = 5.0
BAD_SNR_STD_DB = 4.0

# Initial state prior (no observations yet): assume good until shown otherwise.
INITIAL_P_GOOD = 0.9
