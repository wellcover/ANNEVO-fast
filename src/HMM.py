# -*- coding: utf-8 -*-
"""HMM.py — gene-structure HMM with a sparse-edge Viterbi decoder.

The state machine encodes eukaryotic gene grammar: phase-tracked CDS states
(CDS0/1/2 with frame carry-over across introns), splice donor/acceptor motif
helper states (GT-AG canonical, GC-AG penalized, optional AT-AC U12-type path),
start/stop codon states and intergenic state. Transitions are conditioned on
the current nucleotide (A/T/C/G/N matrices).

This implementation replaces the dense O(L*S^2) Viterbi inner loop with a
sparse edge table, O(L*E), where E is the number of grammar edges (a few
hundred vs S^2 ~ 29k comparisons per base at min_intron_length=20). Edges are
sorted by (to, from) ascending and evaluated in the same order as the dense
reference — with strict-greater comparison and lowest-index tie-breaking — so
the decoded path is identical to the dense implementation, bit for bit.

State/transition tables and the compiled edge table are cached per process
keyed by (min_intron_length, expect_exon, expect_intron, gcag, atac).
"""
import numpy as np
from numba import njit


def define_state(min_intron_length, atac=False):
    states = [
        'intergenic',
        'start0', 'start1', 'start2',
        'CDS0', 'CDS0_T', 'CDS1', 'CDS1_TA', 'CDS1_TG', 'CDS2',
        'DSS0', 'DSS1', 'DSS2', 'DSS0_T', 'DSS1_TA', 'DSS1_TG',
        'ASS0', 'ASS1', 'ASS2', 'ASS0_T', 'ASS1_TA', 'ASS1_TG',
        'end0', 'end1_TA', 'end1_TG', 'end2',
        'intron0_splice0', 'intron0_splice1', 'intron0_splice2', 'intron0_splice3',
        'intron0_T_splice0', 'intron0_T_splice1', 'intron0_T_splice2', 'intron0_T_splice3',
        'intron1_splice0', 'intron1_splice1', 'intron1_splice2', 'intron1_splice3',
        'intron1_TA_splice0', 'intron1_TA_splice1', 'intron1_TA_splice2', 'intron1_TA_splice3',
        'intron1_TG_splice0', 'intron1_TG_splice1', 'intron1_TG_splice2', 'intron1_TG_splice3',
        'intron2_splice0', 'intron2_splice1', 'intron2_splice2', 'intron2_splice3',
    ]

    if atac:
        # AT-AC (U12-type) minor splice helper states: donor preceded by A/T,
        # acceptor ending in C. Phase stop-codon carry-over is approximated away
        # on the AT-AC path (~0.1% of introns).
        for p in (0, 1, 2):
            states.append(f'intron{p}_ATAC_splice0')
            states.append(f'intron{p}_ATAC_splice1')
        for p in (0, 1, 2):
            states.append(f'intron{p}_ATAC_splice3')

    for i in range(min_intron_length):
        states.append(f'intron0_{i}')
        states.append(f'intron0_T_{i}')
        states.append(f'intron1_{i}')
        states.append(f'intron1_TA_{i}')
        states.append(f'intron1_TG_{i}')
        states.append(f'intron2_{i}')

    states_to_num = {s: i for i, s in enumerate(states)}
    num_states = len(states_to_num)
    return states_to_num, num_states


def define_columns(states_to_num):
    column_groups = {
        'INTERGENIC': ['intergenic'],
        'CODING_EXON_0': ['CDS0', 'CDS0_T'],
        'CODING_EXON_1': ['CDS1', 'CDS1_TA', 'CDS1_TG'],
        'CODING_EXON_2': ['CDS2'],
        'INTRON_0': [state for state in states_to_num if state.startswith('intron0_')],
        'INTRON_1': [state for state in states_to_num if state.startswith('intron1_')],
        'INTRON_2': [state for state in states_to_num if state.startswith('intron2_')],
        'DSS_0': ['DSS0', 'DSS0_T'],
        'DSS_1': ['DSS1', 'DSS1_TA', 'DSS1_TG'],
        'DSS_2': ['DSS2'],
        'ASS_0': ['ASS0', 'ASS0_T'],
        'ASS_1': ['ASS1', 'ASS1_TA', 'ASS1_TG'],
        'ASS_2': ['ASS2'],
        'START': ['start0', 'start1', 'start2'],
        'END': ['end0', 'end1_TA', 'end1_TG', 'end2'],
    }

    column_dict = {}
    for column_name, state_names in column_groups.items():
        column_dict[column_name] = [states_to_num[state] for state in state_names]

    return column_dict


def set_transition_matrix_conditional_state(init_transition_matrix, states_to_num, min_intron_length,
                                            exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty,
                                            intron_quit_penalty,
                                            gcag_penalty_extra=10.0, atac_extra=None):
    transition_matrix_A = init_transition_matrix.copy()
    transition_matrix_G = init_transition_matrix.copy()
    transition_matrix_C = init_transition_matrix.copy()
    transition_matrix_T = init_transition_matrix.copy()
    transition_matrix_other = init_transition_matrix.copy()

    # ---------------------------------- current base == A ----------------------------------
    # CDS-related
    transition_matrix_A[states_to_num[f'intergenic'], states_to_num[f'start0']] = 0
    transition_matrix_A[states_to_num[f'end0'], states_to_num[f'end1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'end1_TA'], states_to_num[f'end2']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'end1_TG'], states_to_num[f'end2']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'start2'], states_to_num[f'DSS0']] = exon_sustain_penalty

    transition_matrix_A[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['CDS2'], states_to_num['DSS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS2'], states_to_num['DSS0']] = exon_sustain_penalty

    transition_matrix_A[states_to_num['CDS0_T'], states_to_num['CDS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['CDS0_T'], states_to_num['DSS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS0_T'], states_to_num['CDS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS0_T'], states_to_num['DSS1_TA']] = exon_sustain_penalty

    # intron-related
    transition_matrix_A[states_to_num[f'intron0_{min_intron_length - 1}'], states_to_num[f'intron0_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron0_T_{min_intron_length - 1}'], states_to_num[f'intron0_T_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron1_{min_intron_length - 1}'], states_to_num[f'intron1_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron1_TA_{min_intron_length - 1}'], states_to_num[f'intron1_TA_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron1_TG_{min_intron_length - 1}'], states_to_num[f'intron1_TG_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron2_{min_intron_length - 1}'], states_to_num[f'intron2_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1_TA']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0']] = intron_quit_penalty

    # AT-AC donor entry (first base A) and acceptor exit
    if atac_extra is not None:
        w_atac_don = exon_quit_penalty - atac_extra
        transition_matrix_A[states_to_num['DSS0'], states_to_num['intron0_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['DSS0_T'], states_to_num['intron0_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['DSS1'], states_to_num['intron1_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['DSS1_TA'], states_to_num['intron1_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['DSS1_TG'], states_to_num['intron1_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['DSS2'], states_to_num['intron2_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['start2'], states_to_num['intron2_ATAC_splice0']] = w_atac_don
        transition_matrix_A[states_to_num['intron0_ATAC_splice3'], states_to_num['ASS1']] = intron_quit_penalty
        transition_matrix_A[states_to_num['intron1_ATAC_splice3'], states_to_num['ASS2']] = intron_quit_penalty
        transition_matrix_A[states_to_num['intron2_ATAC_splice3'], states_to_num['ASS0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == T ----------------------------------
    # CDS-related
    transition_matrix_T[states_to_num[f'start0'], states_to_num[f'start1']] = 0
    transition_matrix_T[states_to_num[f'start2'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'start2'], states_to_num[f'DSS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS2'], states_to_num['end0']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS2'], states_to_num['end0']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS2'], states_to_num['CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS2'], states_to_num['DSS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS2'], states_to_num['CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS2'], states_to_num['DSS0_T']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS0_T'], states_to_num['DSS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS0_T'], states_to_num['DSS1']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty

    # intron_related
    transition_matrix_T[states_to_num['intron0_splice0'], states_to_num['intron0_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron0_T_splice0'], states_to_num['intron0_T_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron1_splice0'], states_to_num['intron1_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron1_TG_splice0'], states_to_num['intron1_TG_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron1_TA_splice0'], states_to_num['intron1_TA_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron2_splice0'], states_to_num['intron2_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron1_TA_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron1_TG_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0_T']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron2_splice3'], states_to_num[f'end0']] = intron_quit_penalty

    # AT-AC donor second base T and acceptor exit
    if atac_extra is not None:
        for p in (0, 1, 2):
            transition_matrix_T[states_to_num[f'intron{p}_ATAC_splice0'],
                                states_to_num[f'intron{p}_ATAC_splice1']] = intron_sustain_penalty
        transition_matrix_T[states_to_num['intron0_ATAC_splice3'], states_to_num['ASS1']] = intron_quit_penalty
        transition_matrix_T[states_to_num['intron1_ATAC_splice3'], states_to_num['ASS2']] = intron_quit_penalty
        transition_matrix_T[states_to_num['intron2_ATAC_splice3'], states_to_num['ASS0_T']] = intron_quit_penalty
        transition_matrix_T[states_to_num['intron2_ATAC_splice3'], states_to_num['end0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == G ----------------------------------
    # CDS-related
    transition_matrix_G[states_to_num[f'start1'], states_to_num[f'start2']] = 0
    transition_matrix_G[states_to_num[f'end0'], states_to_num[f'end1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'end1_TA'], states_to_num[f'end2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'start2'], states_to_num[f'DSS0']] = exon_sustain_penalty

    transition_matrix_G[states_to_num['CDS0_T'], states_to_num['CDS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS0_T'], states_to_num['CDS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS0_T'], states_to_num['DSS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS0_T'], states_to_num['DSS1_TG']] = exon_sustain_penalty

    transition_matrix_G[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty

    transition_matrix_G[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS2'], states_to_num['DSS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS2'], states_to_num['DSS0']] = exon_sustain_penalty

    # intron_related
    transition_matrix_G[states_to_num['DSS0'], states_to_num['intron0_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS0_T'], states_to_num['intron0_T_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS1'], states_to_num['intron1_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS1_TG'], states_to_num['intron1_TG_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS1_TA'], states_to_num['intron1_TA_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS2'], states_to_num['intron2_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['start2'], states_to_num['intron2_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['intron0_splice2'], states_to_num['intron0_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron0_T_splice2'], states_to_num['intron0_T_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron1_splice2'], states_to_num['intron1_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron1_TG_splice2'], states_to_num['intron1_TG_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron1_TA_splice2'], states_to_num['intron1_TA_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron2_splice2'], states_to_num['intron2_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1_TG']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron1_TG_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0']] = intron_quit_penalty

    # AT-AC acceptor exit
    if atac_extra is not None:
        transition_matrix_G[states_to_num['intron0_ATAC_splice3'], states_to_num['ASS1']] = intron_quit_penalty
        transition_matrix_G[states_to_num['intron1_ATAC_splice3'], states_to_num['ASS2']] = intron_quit_penalty
        transition_matrix_G[states_to_num['intron2_ATAC_splice3'], states_to_num['ASS0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == C ----------------------------------
    # CDS-related
    transition_matrix_C[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num[f'start2'], states_to_num[f'DSS0']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS2'], states_to_num['DSS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS2'], states_to_num['DSS0']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS0_T'], states_to_num['DSS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS0_T'], states_to_num['DSS1']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty

    # intron-related
    # GC-AG donor: second base C carries an extra penalty (default 10, historical behaviour)
    transition_matrix_C[states_to_num['intron0_splice0'], states_to_num['intron0_splice1']] = intron_sustain_penalty - gcag_penalty_extra
    transition_matrix_C[states_to_num['intron0_T_splice0'], states_to_num['intron0_T_splice1']] = intron_sustain_penalty - gcag_penalty_extra
    transition_matrix_C[states_to_num['intron1_splice0'], states_to_num['intron1_splice1']] = intron_sustain_penalty - gcag_penalty_extra
    transition_matrix_C[states_to_num['intron1_TG_splice0'], states_to_num['intron1_TG_splice1']] = intron_sustain_penalty - gcag_penalty_extra
    transition_matrix_C[states_to_num['intron1_TA_splice0'], states_to_num['intron1_TA_splice1']] = intron_sustain_penalty - gcag_penalty_extra
    transition_matrix_C[states_to_num['intron2_splice0'], states_to_num['intron2_splice1']] = intron_sustain_penalty - gcag_penalty_extra

    transition_matrix_C[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron1_TA_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron1_TG_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0']] = intron_quit_penalty

    # AT-AC acceptor: after the splice2 A, a C (not G) enters ATAC_splice3; acceptor exit
    if atac_extra is not None:
        for p in (0, 1, 2):
            transition_matrix_C[states_to_num[f'intron{p}_splice2'],
                                states_to_num[f'intron{p}_ATAC_splice3']] = intron_sustain_penalty - atac_extra
        transition_matrix_C[states_to_num['intron0_ATAC_splice3'], states_to_num['ASS1']] = intron_quit_penalty
        transition_matrix_C[states_to_num['intron1_ATAC_splice3'], states_to_num['ASS2']] = intron_quit_penalty
        transition_matrix_C[states_to_num['intron2_ATAC_splice3'], states_to_num['ASS0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == N ----------------------------------
    # Open any ATCG-conditional path under 'N/other' with a fixed penalty.
    atcg_open_mask = (
        np.isfinite(transition_matrix_A) |
        np.isfinite(transition_matrix_T) |
        np.isfinite(transition_matrix_G) |
        np.isfinite(transition_matrix_C)
    )
    conditional_open_mask = atcg_open_mask & (~np.isfinite(init_transition_matrix))
    transition_matrix_other[conditional_open_mask] = -10
    # ---------------------------------------------------------------------------------------

    transition_matrix_dict = {
        'A': transition_matrix_A,
        'T': transition_matrix_T,
        'C': transition_matrix_C,
        'G': transition_matrix_G,
        'N': transition_matrix_other,
    }
    return transition_matrix_dict


def set_transition_matrix_common_state(transition_matrix, states_to_num, min_intron_length,
                                       exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty,
                                       intron_quit_penalty):
    transition_matrix[states_to_num['intergenic'], states_to_num['intergenic']] = 0
    transition_matrix[states_to_num['end2'], states_to_num['intergenic']] = exon_quit_penalty

    transition_matrix[states_to_num['CDS0'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS0'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS0'], states_to_num['DSS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS0'], states_to_num['DSS1']] = exon_sustain_penalty

    transition_matrix[states_to_num['CDS1'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS1'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS1'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS1'], states_to_num['DSS2']] = exon_sustain_penalty

    # Bridge donor motif helper states back to intron bodies.
    transition_matrix[states_to_num['intron0_splice1'], states_to_num['intron0_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron0_T_splice1'], states_to_num['intron0_T_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron1_splice1'], states_to_num['intron1_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron1_TA_splice1'], states_to_num['intron1_TA_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron1_TG_splice1'], states_to_num['intron1_TG_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron2_splice1'], states_to_num['intron2_0']] = intron_sustain_penalty

    # AT-AC donor helper states bridge back to intron bodies
    if 'intron0_ATAC_splice1' in states_to_num:
        for p in (0, 1, 2):
            transition_matrix[states_to_num[f'intron{p}_ATAC_splice1'],
                              states_to_num[f'intron{p}_0']] = intron_sustain_penalty

    transition_matrix[states_to_num[f'intron0_0'], states_to_num[f'intron0_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron0_T_0'], states_to_num[f'intron0_T_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron1_0'], states_to_num[f'intron1_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron1_TA_0'], states_to_num[f'intron1_TA_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron1_TG_0'], states_to_num[f'intron1_TG_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron2_0'], states_to_num[f'intron2_0']] = intron_sustain_penalty
    if min_intron_length > 1:
        for i in range(min_intron_length - 1):
            transition_matrix[states_to_num[f'intron0_{i}'], states_to_num[f'intron0_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron0_T_{i}'], states_to_num[f'intron0_T_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron1_{i}'], states_to_num[f'intron1_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron1_TA_{i}'], states_to_num[f'intron1_TA_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron1_TG_{i}'], states_to_num[f'intron1_TG_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron2_{i}'], states_to_num[f'intron2_{i + 1}']] = intron_sustain_penalty

    return transition_matrix


# ===========================================================================
# Sparse-edge Viterbi core (numba-jitted, float32 / float64 variants).
#
# Edge arrays are globally concatenated in (to, from) ascending order;
# to_ptr[sym, j] gives the edge interval of target state j under symbol sym.
# Candidates are evaluated as ((dp_prev + w) + emit) with strict-greater
# comparison; from-ascending order makes ties resolve to the smallest from
# index, so the result matches the dense reference implementation exactly.
# ===========================================================================
@njit(cache=True)
def _viterbi_core_sparse_f32(log_emit_probs, edge_from, edge_weight, to_ptr, sequence_codes):
    seq_length, num_states = log_emit_probs.shape
    path = np.zeros((seq_length, num_states), dtype=np.int32)
    dp = np.full((seq_length, num_states), -np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for t in range(1, seq_length):
        sym = sequence_codes[t]
        for to_state in range(num_states):
            best_score = -np.inf
            best_from = 0
            emit_score = log_emit_probs[t, to_state]
            for e in range(to_ptr[sym, to_state], to_ptr[sym, to_state + 1]):
                score = dp[t - 1, edge_from[e]] + edge_weight[e] + emit_score
                if score > best_score:
                    best_score = score
                    best_from = edge_from[e]
            dp[t, to_state] = best_score
            path[t, to_state] = best_from

    best_path = np.empty(seq_length, dtype=np.int32)
    best_path[seq_length - 1] = 0
    for t in range(seq_length - 1, 0, -1):
        best_path[t - 1] = path[t, best_path[t]]
    return best_path


@njit(cache=True)
def _viterbi_core_sparse_f64(log_emit_probs, edge_from, edge_weight, to_ptr, sequence_codes):
    seq_length, num_states = log_emit_probs.shape
    path = np.zeros((seq_length, num_states), dtype=np.int32)
    dp = np.full((seq_length, num_states), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for t in range(1, seq_length):
        sym = sequence_codes[t]
        for to_state in range(num_states):
            best_score = -np.inf
            best_from = 0
            emit_score = log_emit_probs[t, to_state]
            for e in range(to_ptr[sym, to_state], to_ptr[sym, to_state + 1]):
                score = dp[t - 1, edge_from[e]] + edge_weight[e] + emit_score
                if score > best_score:
                    best_score = score
                    best_from = edge_from[e]
            dp[t, to_state] = best_score
            path[t, to_state] = best_from

    best_path = np.empty(seq_length, dtype=np.int32)
    best_path[seq_length - 1] = 0
    for t in range(seq_length - 1, 0, -1):
        best_path[t - 1] = path[t, best_path[t]]
    return best_path


# ---------------------------------------------------------------------------
# Vectorized base encoding: non-ATCG characters map to 4 (=N)
# ---------------------------------------------------------------------------
_BASE_CODE_LUT = np.full(256, 4, dtype=np.int32)
for _b, _c in ((ord('A'), 0), (ord('T'), 1), (ord('C'), 2), (ord('G'), 3)):
    _BASE_CODE_LUT[_b] = _c


def _encode_sequence(sequence):
    raw = sequence.encode('ascii') if isinstance(sequence, str) else sequence
    return _BASE_CODE_LUT[np.frombuffer(raw, dtype=np.uint8)]


# ---------------------------------------------------------------------------
# Process-level cache of state tables / transition matrices / edge tables
# ---------------------------------------------------------------------------
_HMM_ARTIFACTS_CACHE = {}


def _build_hmm_artifacts(min_intron_length, expect_exon, expect_intron,
                         gcag_penalty_extra=10.0, atac_extra=None):
    atac = atac_extra is not None
    states_to_num, num_states = define_state(min_intron_length, atac=atac)
    columns_dict = define_columns(states_to_num)

    if expect_exon:
        exon_sustain_penalty = np.log(1 - 1 / expect_exon)
        exon_quit_penalty = np.log(1 / expect_exon)
    else:
        exon_sustain_penalty = 0
        exon_quit_penalty = 0
    if expect_intron:
        intron_sustain_penalty = np.log(1 - 1 / expect_intron)
        intron_quit_penalty = np.log(1 / expect_intron)
    else:
        intron_sustain_penalty = 0
        intron_quit_penalty = 0

    init_transition_matrix = np.full((num_states, num_states), -np.inf, dtype=np.float32)
    init_transition_matrix = set_transition_matrix_common_state(init_transition_matrix, states_to_num,
                                                                min_intron_length,
                                                                exon_sustain_penalty, exon_quit_penalty,
                                                                intron_sustain_penalty, intron_quit_penalty)
    transition_matrix_dict = set_transition_matrix_conditional_state(init_transition_matrix,
                                                                     states_to_num, min_intron_length,
                                                                     exon_sustain_penalty,
                                                                     exon_quit_penalty,
                                                                     intron_sustain_penalty,
                                                                     intron_quit_penalty,
                                                                     gcag_penalty_extra=gcag_penalty_extra,
                                                                     atac_extra=atac_extra)
    # Symbol order matches _encode_sequence: A=0, T=1, C=2, G=3, N=4
    dense = np.stack([
        transition_matrix_dict['A'],
        transition_matrix_dict['T'],
        transition_matrix_dict['C'],
        transition_matrix_dict['G'],
        transition_matrix_dict['N'],
    ]).astype(np.float32)

    edge_from_parts, edge_w_parts, to_ptr_rows = [], [], []
    offset = 0
    for sym in range(5):
        M = dense[sym]
        fr, to = np.nonzero(np.isfinite(M))
        w = M[fr, to]
        order = np.lexsort((fr, to))          # primary key: to ascending; secondary: from
        fr, to, w = fr[order], to[order], w[order]
        counts = np.bincount(to, minlength=num_states)
        block_ptr = np.concatenate(([0], np.cumsum(counts)))
        edge_from_parts.append(fr.astype(np.int32))
        edge_w_parts.append(w.astype(np.float32))
        to_ptr_rows.append(offset + block_ptr)
        offset += len(fr)

    artifacts = {
        'states_to_num': states_to_num,
        'num_states': num_states,
        'columns_dict': columns_dict,
        'edge_from': np.concatenate(edge_from_parts),
        'edge_w': np.concatenate(edge_w_parts),
        'to_ptr': np.stack(to_ptr_rows).astype(np.int64),
        'n_edges': offset,
    }
    return artifacts


def _get_hmm_artifacts(min_intron_length, expect_exon=None, expect_intron=None,
                       gcag_penalty_extra=10.0, atac_extra=None):
    key = (min_intron_length, expect_exon, expect_intron, float(gcag_penalty_extra), atac_extra)
    artifacts = _HMM_ARTIFACTS_CACHE.get(key)
    if artifacts is None:
        artifacts = _build_hmm_artifacts(min_intron_length, expect_exon, expect_intron,
                                         gcag_penalty_extra=gcag_penalty_extra, atac_extra=atac_extra)
        _HMM_ARTIFACTS_CACHE[key] = artifacts
    return artifacts


def get_state_tables(min_intron_length, gcag_penalty_extra=10.0, atac_extra=None):
    """Query the cached state tables (shared by the decoding side).

    gcag_penalty_extra / atac_extra must match the values passed to
    viterbi_decoding, otherwise state count and emission columns get out of
    sync.
    """
    artifacts = _get_hmm_artifacts(min_intron_length, gcag_penalty_extra=gcag_penalty_extra,
                                   atac_extra=atac_extra)
    return artifacts['states_to_num'], artifacts['num_states'], artifacts['columns_dict']


def viterbi_decoding(predictions, sequence, states_to_num, num_states, columns_dict, min_intron_length,
                     expect_exon=None, expect_intron=None, extra_penalty=None,
                     gcag_penalty_extra=10.0, atac_extra=None):
    """
    Decode gene structure with the Viterbi algorithm (sparse-edge implementation).

    predictions: [L, 15] per-base class probabilities (softmax output of an
        ANNEVO-style per-base model).
    gcag_penalty_extra: extra log penalty of GC-AG donors relative to canonical
        GT (default 10.0, historical behaviour).
    atac_extra: enables the AT-AC splice path with this extra log penalty;
        None (default) disables it, keeping the state machine identical to the
        historical version.
    Returns the best state path as np.ndarray of shape [L].
    """
    np.seterr(divide='ignore', invalid='ignore')
    epsilon = 1e-3
    predictions[predictions < epsilon] = epsilon
    seq_length = predictions.shape[0]
    artifacts = _get_hmm_artifacts(min_intron_length, expect_exon, expect_intron,
                                   gcag_penalty_extra=gcag_penalty_extra, atac_extra=atac_extra)

    # Emission probabilities: 15 prediction columns mapped onto state groups, float32
    log_emit_probs = np.zeros((seq_length, artifacts['num_states']), dtype=np.float32)

    log_emit_probs[:, columns_dict['INTERGENIC']] = np.log(predictions[:, 0][:, np.newaxis])
    log_emit_probs[:, columns_dict['CODING_EXON_0']] = np.log(predictions[:, 1][:, np.newaxis])
    log_emit_probs[:, columns_dict['CODING_EXON_1']] = np.log(predictions[:, 3][:, np.newaxis])
    log_emit_probs[:, columns_dict['CODING_EXON_2']] = np.log(predictions[:, 2][:, np.newaxis])
    log_emit_probs[:, columns_dict['INTRON_0']] = np.log(predictions[:, 4][:, np.newaxis])
    log_emit_probs[:, columns_dict['INTRON_1']] = np.log(predictions[:, 6][:, np.newaxis])
    log_emit_probs[:, columns_dict['INTRON_2']] = np.log(predictions[:, 5][:, np.newaxis])
    log_emit_probs[:, columns_dict['DSS_0']] = np.log(predictions[:, 7][:, np.newaxis])
    log_emit_probs[:, columns_dict['DSS_1']] = np.log(predictions[:, 9][:, np.newaxis])
    log_emit_probs[:, columns_dict['DSS_2']] = np.log(predictions[:, 8][:, np.newaxis])
    log_emit_probs[:, columns_dict['ASS_0']] = np.log(predictions[:, 10][:, np.newaxis])
    log_emit_probs[:, columns_dict['ASS_1']] = np.log(predictions[:, 12][:, np.newaxis])
    log_emit_probs[:, columns_dict['ASS_2']] = np.log(predictions[:, 11][:, np.newaxis])
    log_emit_probs[:, columns_dict['START']] = np.log(predictions[:, 13][:, np.newaxis])
    log_emit_probs[:, columns_dict['END']] = np.log(predictions[:, 14][:, np.newaxis])

    sequence_codes = _encode_sequence(sequence)
    if seq_length <= 1_000_000:
        best_path = _viterbi_core_sparse_f32(
            log_emit_probs,
            artifacts['edge_from'],
            artifacts['edge_w'],
            artifacts['to_ptr'],
            sequence_codes,
        )
    else:
        best_path = _viterbi_core_sparse_f64(
            log_emit_probs,
            artifacts['edge_from'],
            artifacts['edge_w'],
            artifacts['to_ptr'],
            sequence_codes,
        )
    return best_path
