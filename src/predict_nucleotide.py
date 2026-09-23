# -*- coding: utf-8 -*-
"""Minimal nucleotide utilities for the decoding package."""


def rev_complement(dna_sequence):
    complement_map = str.maketrans('ATGCatgcNXnx', 'TACGtacgNXnx')
    return dna_sequence.translate(complement_map)[::-1]
