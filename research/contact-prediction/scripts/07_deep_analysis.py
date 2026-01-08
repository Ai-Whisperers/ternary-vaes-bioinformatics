#!/usr/bin/env python3
"""
DEEP ANALYSIS: Multiple hypotheses testing for codon-physics encoding.

Tests:
1. Local vs long-range contacts (|i-j| ranges)
2. Secondary structure elements (helix-helix, sheet-sheet, helix-sheet)
3. Correlation with actual folding rates from literature
4. Contact density effects
5. Hydrophobic vs polar contacts
6. Expanded protein set (25+ proteins)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance

# =============================================================================
# EXPANDED PROTEIN DATABASE WITH FOLDING RATES
# =============================================================================

# Folding rates from literature (ln(kf) in s^-1, or qualitative)
# Sources: Plaxco et al., Maxwell et al., various NMR/kinetics studies

PROTEINS = {
    # =========================================================================
    # ULTRAFAST FOLDERS (τ < 10 μs, ln(kf) > 11.5)
    # =========================================================================
    'chignolin': {
        'name': 'Chignolin',
        'length': 10,
        'pdb': '1UAO',
        'fold_type': 'beta',
        'constraint': 'designed',
        'folding_rate': 13.8,  # ~1 μs
        'sequence': 'GYDPETGTWG',
        'codons': ['GGC', 'TAC', 'GAC', 'CCC', 'GAG', 'ACC', 'GGC', 'ACC', 'TGG', 'GGC'],
        'ca_coords': np.array([
            [1.458, -0.517, 0.034], [2.634, 2.951, -0.466], [0.193, 4.825, 1.587],
            [-2.261, 2.235, 1.667], [-0.718, -0.929, 2.792], [2.580, -1.047, 4.513],
            [2.046, 2.562, 5.274], [-1.318, 3.002, 6.680], [-2.826, -0.360, 6.042],
            [-0.029, -2.544, 5.221],
        ]),
        'ss': 'CCEEEEEECC',  # Secondary structure
    },

    'trp_cage': {
        'name': 'Trp-cage TC5b',
        'length': 20,
        'pdb': '1L2Y',
        'fold_type': 'alpha',
        'constraint': 'designed',
        'folding_rate': 12.4,  # ~4 μs
        'sequence': 'NLYIQWLKDGGPSSGRPPPS',
        'codons': ['AAC', 'CTG', 'TAC', 'ATC', 'CAG', 'TGG', 'CTG', 'AAG', 'GAC', 'GGC',
                   'GGC', 'CCC', 'AGC', 'AGC', 'GGC', 'CGC', 'CCC', 'CCC', 'CCC', 'AGC'],
        'ca_coords': np.array([
            [-8.284, 1.757, 5.847], [-7.665, 5.476, 5.618], [-7.853, 6.821, 2.090],
            [-5.306, 9.318, 1.063], [-2.566, 7.447, 2.796], [-3.928, 5.040, 5.449],
            [-2.101, 2.107, 4.109], [-4.139, -0.706, 2.667], [-1.792, -2.684, 0.772],
            [-2.665, -2.155, -2.905], [0.376, -0.123, -3.516], [0.596, 3.055, -1.541],
            [4.198, 2.212, -0.674], [4.858, 0.133, 2.315], [2.596, -2.783, 3.077],
            [4.756, -5.428, 1.481], [2.903, -7.115, -1.263], [3.697, -4.619, -4.050],
            [0.030, -4.701, -5.171], [-1.389, -1.197, -4.795],
        ]),
        'ss': 'CHHHHHHHHCCCCCCCCPPP',
    },

    'villin_hp35': {
        'name': 'Villin HP35',
        'length': 35,
        'pdb': '1YRF',
        'fold_type': 'alpha',
        'constraint': 'hydrophobic',
        'folding_rate': 12.3,  # ~4.3 μs - one of fastest natural proteins
        'sequence': 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
        'codons': ['CTG', 'AGC', 'GAC', 'GAG', 'GAC', 'TTC', 'AAG', 'GCC', 'GTG', 'TTC',
                   'GGC', 'ATG', 'ACC', 'CGC', 'AGC', 'GCC', 'TTC', 'GCC', 'AAC', 'CTG',
                   'CCC', 'CTG', 'TGG', 'AAG', 'CAG', 'CAG', 'AAC', 'CTG', 'AAG', 'AAG',
                   'GAG', 'AAG', 'GGC', 'CTG', 'TTC'],
        'ca_coords': np.array([
            [24.040, 19.584, 7.913], [23.093, 16.012, 8.580], [21.893, 14.823, 11.916],
            [18.188, 14.576, 11.626], [16.478, 11.229, 11.717], [14.128, 11.023, 14.603],
            [10.590, 10.021, 13.689], [9.627, 12.555, 11.074], [12.065, 14.904, 9.689],
            [10.635, 17.422, 7.345], [13.310, 19.950, 6.723], [12.397, 22.396, 4.085],
            [14.942, 24.952, 3.183], [13.723, 27.471, 0.712], [16.026, 29.730, -1.179],
            [14.413, 31.439, -4.179], [16.692, 34.418, -4.496], [14.824, 37.603, -4.890],
            [16.880, 39.377, -7.493], [14.510, 41.978, -8.829], [15.954, 43.064, -12.201],
            [13.017, 45.361, -13.041], [13.588, 45.620, -16.778], [10.275, 47.360, -17.655],
            [9.853, 46.574, -21.330], [6.279, 47.700, -22.106], [5.159, 45.917, -25.232],
            [1.579, 46.986, -25.524], [-0.082, 44.262, -27.531], [-3.541, 45.339, -28.525],
            [-5.344, 42.343, -30.076], [-8.954, 43.210, -30.711], [-10.794, 40.096, -31.757],
            [-14.398, 40.673, -32.561], [-15.983, 37.345, -33.315],
        ]),
        'ss': 'CHHHHHHHHHCCHHHHHHHHHCHHHHHHHHHHHHHH',
    },

    'ww_fbp28': {
        'name': 'WW Domain FBP28',
        'length': 37,
        'pdb': '1E0L',
        'fold_type': 'beta',
        'constraint': 'hydrophobic',
        'folding_rate': 11.7,  # ~13 μs - ultrafast beta
        'sequence': 'GSKLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG',
        'codons': ['GGC', 'AGC', 'AAG', 'CTG', 'CCC', 'CCC', 'GGC', 'TGG', 'GAG', 'AAG',
                   'CGC', 'ATG', 'AGC', 'CGC', 'AGC', 'AGC', 'GGC', 'CGC', 'GTG', 'TAC',
                   'TAC', 'TTC', 'AAC', 'CAC', 'ATC', 'ACC', 'AAC', 'GCC', 'AGC', 'CAG',
                   'TTC', 'GAG', 'CGC', 'CCC', 'AGC', 'GGC', 'GGC'],
        'ca_coords': np.array([
            [4.795, 13.025, 22.117], [2.584, 10.086, 22.336], [0.885, 9.398, 18.903],
            [-2.117, 7.180, 19.155], [-3.544, 7.145, 15.598], [-6.651, 4.898, 15.488],
            [-7.806, 4.742, 11.870], [-10.715, 2.453, 11.530], [-11.569, 2.095, 7.827],
            [-14.285, -0.445, 7.437], [-14.884, -0.970, 3.708], [-17.412, -3.769, 3.284],
            [-17.785, -4.554, -0.426], [-20.110, -7.490, -0.753], [-20.289, -8.534, -4.424],
            [-22.316, -11.633, -4.679], [-22.199, -12.920, -8.231], [-24.012, -16.145, -8.457],
            [-23.567, -17.697, -11.884], [-25.100, -21.129, -12.051], [-24.331, -22.853, -15.354],
            [-25.613, -26.377, -15.390], [-24.576, -28.266, -18.500], [-25.571, -31.870, -18.365],
            [-24.225, -33.902, -21.292], [-24.944, -37.585, -21.019], [-23.349, -39.730, -23.746],
            [-23.806, -43.418, -23.218], [-21.913, -45.615, -25.790], [-21.967, -49.261, -24.883],
            [-19.821, -51.428, -27.305], [-19.529, -55.029, -26.226], [-17.040, -56.994, -28.468],
            [-16.420, -60.614, -27.477], [-13.665, -62.307, -29.606], [-12.671, -65.838, -28.596],
            [-9.763, -67.213, -30.631],
        ]),
        'ss': 'CCCCCCCEEEEECCCCEEEEEECCCEEEEEECCCCCC',
    },

    # =========================================================================
    # FAST FOLDERS (τ ~ 10 μs - 1 ms, ln(kf) 7-11.5)
    # =========================================================================
    'protein_l': {
        'name': 'Protein L',
        'length': 62,
        'pdb': '1HZ6',
        'fold_type': 'alpha/beta',
        'constraint': 'hydrophobic',
        'folding_rate': 9.2,  # ~100 μs
        'sequence': 'MEEVTIKANLIFANGSTQTAEFKGTFEKATSEAYAYADTLKKDNGEWTVDVADKGYTLNIKFAG',
        'codons': ['ATG', 'GAG', 'GAG', 'GTG', 'ACC', 'ATC', 'AAG', 'GCC', 'AAC', 'CTG',
                   'ATC', 'TTC', 'GCC', 'AAC', 'GGC', 'AGC', 'ACC', 'CAG', 'ACC', 'GCC',
                   'GAG', 'TTC', 'AAG', 'GGC', 'ACC', 'TTC', 'GAG', 'AAG', 'GCC', 'ACC',
                   'AGC', 'GAG', 'GCC', 'TAC', 'GCC', 'TAC', 'GCC', 'GAC', 'ACC', 'CTG',
                   'AAG', 'AAG', 'GAC', 'AAC', 'GGC', 'GAG', 'TGG', 'ACC', 'GTG', 'GAC',
                   'GTG', 'GCC', 'GAC', 'AAG', 'GGC', 'TAC', 'ACC', 'CTG', 'AAC', 'ATC',
                   'AAG', 'TTC'],
        'ca_coords': np.array([
            [29.285, 2.000, 12.689], [26.017, 3.832, 13.270], [24.012, 1.124, 11.693],
            [21.146, 3.392, 11.099], [18.777, 0.662, 9.983], [16.245, 2.728, 8.430],
            [13.420, 0.290, 7.860], [11.395, 2.635, 5.709], [8.306, 0.645, 5.396],
            [6.624, 2.840, 2.744], [3.465, 1.022, 2.609], [2.134, 3.231, -0.240],
            [-1.010, 1.330, -0.625], [-1.979, 3.362, -3.550], [-5.356, 1.792, -4.127],
            [-5.870, 3.649, -7.310], [-9.375, 2.544, -8.019], [-9.469, 4.091, -11.459],
            [-12.867, 2.629, -12.314], [-12.567, 3.656, -15.935], [-15.746, 1.758, -16.773],
            [-15.033, 2.154, -20.470], [-14.968, -1.620, -20.622], [-11.495, -2.666, -21.542],
            [-10.814, -6.400, -21.466], [-7.324, -7.382, -22.432], [-6.427, -11.024, -22.078],
            [-2.894, -11.876, -22.980], [-1.710, -15.356, -22.213], [1.839, -15.921, -23.147],
            [3.256, -19.244, -22.009], [6.813, -19.505, -23.109], [8.439, -22.682, -21.758],
            [11.923, -22.548, -23.046], [13.720, -25.629, -21.719], [17.186, -25.201, -22.921],
            [19.063, -28.171, -21.520], [22.466, -27.471, -22.909], [24.394, -30.228, -21.299],
            [27.728, -29.174, -22.626], [29.584, -31.681, -20.557], [32.876, -30.308, -21.541],
            [34.604, -32.494, -19.033], [37.795, -30.709, -19.562], [39.362, -32.391, -16.601],
            [42.474, -30.285, -16.602], [43.822, -31.305, -13.175], [46.779, -28.849, -13.005],
            [47.834, -29.185, -9.389], [50.548, -26.592, -9.081], [51.235, -26.340, -5.328],
            [53.663, -23.572, -4.879], [53.943, -22.776, -1.152], [56.028, -19.659, -0.609],
            [55.789, -18.235, 2.947], [57.446, -14.859, 3.408], [56.813, -12.818, 6.617],
            [58.044, -9.220, 6.697], [56.879, -6.806, 9.342], [57.693, -3.129, 9.044],
            [56.100, -0.494, 11.438], [56.533, 3.122, 10.507],
        ]),
        'ss': 'CCEEEEEEECCCCCHHHHHHHHHCCCEEEEEECCCCCEEEEEECCCCCHHHHHHHCCEEEEE',
    },

    'src_sh3': {
        'name': 'Src SH3 Domain',
        'length': 57,
        'pdb': '1SRL',
        'fold_type': 'beta',
        'constraint': 'hydrophobic',
        'folding_rate': 8.5,  # ~200 μs
        'sequence': 'GGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLTTGQTGYIPSNYVAPS',
        'codons': ['GGC', 'GGC', 'GTG', 'ACC', 'ACC', 'TTC', 'GTG', 'GCC', 'CTG', 'TAC',
                   'GAC', 'TAC', 'GAG', 'AGC', 'CGC', 'ACC', 'GAG', 'ACC', 'GAC', 'CTG',
                   'AGC', 'TTC', 'AAG', 'AAG', 'GGC', 'GAG', 'CGC', 'CTG', 'CAG', 'ATC',
                   'GTG', 'AAC', 'AAC', 'ACC', 'GAG', 'GGC', 'GAC', 'TGG', 'TGG', 'CTG',
                   'GCC', 'CAC', 'AGC', 'CTG', 'ACC', 'ACC', 'GGC', 'CAG', 'ACC', 'GGC',
                   'TAC', 'ATC', 'CCC', 'AGC', 'AAC', 'TAC', 'GTG'],
        'ca_coords': np.array([
            [10.204, 17.491, 6.523], [11.102, 17.227, 10.186], [8.024, 18.893, 11.650],
            [9.076, 21.440, 14.085], [6.337, 23.883, 14.605], [7.260, 26.434, 17.227],
            [4.681, 29.214, 17.327], [5.455, 32.121, 19.671], [2.914, 34.870, 19.437],
            [3.530, 37.536, 21.934], [1.051, 40.421, 21.582], [1.487, 43.119, 24.148],
            [-1.137, 45.809, 23.657], [-0.891, 48.521, 26.217], [-3.580, 51.147, 25.754],
            [-3.580, 54.020, 28.188], [-5.355, 53.152, 31.397], [-4.008, 50.171, 33.224],
            [-5.989, 47.030, 32.490], [-4.285, 45.085, 29.703], [-6.021, 41.699, 29.165],
            [-3.895, 39.638, 26.772], [-5.300, 36.132, 26.291], [-2.977, 34.196, 24.119],
            [-3.813, 30.515, 23.713], [-1.102, 28.531, 21.796], [-1.467, 24.814, 21.372],
            [1.321, 22.609, 19.862], [1.274, 19.051, 18.574], [4.152, 16.740, 17.476],
            [4.451, 13.193, 16.146], [7.326, 10.926, 15.291], [7.834, 7.461, 13.906],
            [10.706, 5.124, 13.176], [11.342, 1.791, 11.528], [14.253, -0.633, 10.943],
            [14.935, -3.853, 8.989], [17.935, -5.871, 8.352], [18.655, -9.179, 6.582],
            [21.665, -11.149, 5.880], [22.341, -14.466, 4.109], [22.096, -14.220, 0.314],
            [25.143, -12.145, -0.828], [25.030, -10.012, -4.013], [28.481, -8.515, -4.469],
            [29.063, -4.772, -4.717], [32.630, -3.714, -5.425], [33.666, 0.004, -5.339],
            [30.614, 1.915, -5.903], [29.574, 4.456, -8.362], [26.023, 5.442, -7.512],
            [24.563, 8.780, -8.525], [21.040, 9.449, -7.296], [19.232, 12.620, -7.987],
            [15.717, 13.035, -6.793], [13.659, 16.145, -7.244], [10.118, 16.084, -6.010],
        ]),
        'ss': 'CCEEEEEECCCCEEEEEECCCCCCCEEEEEEECCCCEEEEECCCCCEEEEEECCCCCCC',
    },

    'gb1': {
        'name': 'Protein G B1',
        'length': 56,
        'pdb': '1PGB',
        'fold_type': 'alpha/beta',
        'constraint': 'hydrophobic',
        'folding_rate': 7.6,  # ~500 μs
        'sequence': 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
        'codons': ['ATG', 'ACC', 'TAC', 'AAG', 'CTG', 'ATC', 'CTG', 'AAC', 'GGC', 'AAG',
                   'ACC', 'CTG', 'AAG', 'GGC', 'GAG', 'ACC', 'ACC', 'ACC', 'GAG', 'GCC',
                   'GTG', 'GAC', 'GCC', 'GCC', 'ACC', 'GCC', 'GAG', 'AAG', 'GTG', 'TTC',
                   'AAG', 'CAG', 'TAC', 'GCC', 'AAC', 'GAC', 'AAC', 'GGC', 'GTG', 'GAC',
                   'GGC', 'GAG', 'TGG', 'ACC', 'TAC', 'GAC', 'GAC', 'GCC', 'ACC', 'AAG',
                   'ACC', 'TTC', 'ACC', 'GTG', 'ACC', 'GAG'],
        'ca_coords': np.array([
            [16.671, 43.549, 34.916], [17.088, 44.071, 31.201], [15.063, 41.265, 29.947],
            [17.159, 38.243, 30.434], [16.513, 37.621, 34.134], [19.393, 39.947, 34.876],
            [19.109, 42.006, 31.695], [21.927, 39.820, 30.582], [21.461, 36.628, 32.552],
            [23.173, 37.963, 35.680], [26.128, 40.159, 34.749], [25.336, 40.739, 31.062],
            [28.310, 38.562, 30.209], [28.159, 35.259, 32.013], [30.070, 35.997, 35.163],
            [33.039, 38.352, 34.491], [32.314, 39.360, 30.863], [35.369, 37.230, 30.071],
            [35.369, 33.936, 31.926], [37.195, 34.352, 35.202], [40.345, 36.462, 34.571],
            [40.082, 37.491, 30.893], [43.498, 35.894, 30.093], [44.049, 32.569, 31.867],
            [45.680, 32.835, 35.269], [48.939, 34.728, 34.638], [49.233, 35.976, 30.998],
            [52.733, 34.635, 30.354], [53.492, 31.222, 31.738], [55.132, 30.997, 35.202],
            [58.508, 32.710, 34.635], [58.999, 34.228, 31.103], [62.368, 32.666, 30.241],
            [63.355, 29.358, 31.800], [64.904, 29.103, 35.261], [68.293, 30.807, 34.706],
            [68.980, 32.435, 31.247], [72.243, 30.677, 30.362], [73.294, 27.340, 31.880],
            [74.652, 27.163, 35.439], [77.909, 29.096, 34.866], [78.785, 30.631, 31.364],
            [81.823, 28.554, 30.561], [82.802, 25.168, 31.995], [84.169, 24.862, 35.522],
            [87.454, 26.815, 34.964], [88.285, 28.490, 31.553], [91.241, 26.299, 30.669],
            [92.193, 22.972, 32.178], [93.476, 22.612, 35.724], [96.747, 24.570, 35.211],
            [97.648, 26.274, 31.835], [100.605, 24.076, 30.875], [101.586, 20.688, 32.316],
            [102.891, 20.362, 35.880], [106.162, 22.326, 35.363],
        ]),
        'ss': 'CEEEEEECCCCHHHHHHHHCCCCEEEEEECCCCCCCEEEEECCCCHHHHHHCCEEEEE',
    },

    'lambda_repressor': {
        'name': 'Lambda Repressor (6-85)',
        'length': 80,
        'pdb': '1LMB',
        'fold_type': 'alpha',
        'constraint': 'hydrophobic',
        'folding_rate': 9.9,  # ~50 μs (fast for this size)
        'sequence': 'TQEQLEDARRLKAIYEKKKNELGLSQESVADKMGMGQSGVGALFNGINALNAYNAALLAKILKVSVEEFSPSIAREIYEMYEAVSMQPS',
        'codons': ['ACC', 'CAG', 'GAG', 'CAG', 'CTG', 'GAG', 'GAC', 'GCC', 'CGC', 'CGC',
                   'CTG', 'AAG', 'GCC', 'ATC', 'TAC', 'GAG', 'AAG', 'AAG', 'AAG', 'AAC',
                   'GAG', 'CTG', 'GGC', 'CTG', 'AGC', 'CAG', 'GAG', 'AGC', 'GTG', 'GCC',
                   'GAC', 'AAG', 'ATG', 'GGC', 'ATG', 'GGC', 'CAG', 'AGC', 'GGC', 'GTG',
                   'GGC', 'GCC', 'CTG', 'TTC', 'AAC', 'GGC', 'ATC', 'AAC', 'GCC', 'CTG',
                   'AAC', 'GCC', 'TAC', 'AAC', 'GCC', 'GCC', 'CTG', 'CTG', 'GCC', 'AAG',
                   'ATC', 'CTG', 'AAG', 'GTG', 'AGC', 'GTG', 'GAG', 'GAG', 'TTC', 'AGC',
                   'CCC', 'AGC', 'ATC', 'GCC', 'CGC', 'GAG', 'ATC', 'TAC', 'GAG', 'ATG'],
        'ca_coords': np.array([
            [28.566, 51.084, -1.779], [25.842, 49.104, 0.265], [22.561, 50.848, 0.870],
            [19.892, 48.223, 1.236], [16.593, 49.909, 1.961], [13.903, 47.329, 2.141],
            [10.530, 48.936, 2.713], [7.884, 46.296, 2.622], [4.447, 47.652, 3.373],
            [1.867, 44.893, 3.092], [2.019, 44.034, 6.810], [1.553, 47.463, 8.201],
            [4.611, 49.537, 8.232], [4.261, 52.564, 10.450], [7.587, 54.386, 10.308],
            [7.371, 57.345, 12.554], [10.904, 58.846, 12.352], [10.994, 62.145, 14.265],
            [11.008, 61.598, 18.008], [14.461, 63.145, 17.877], [15.236, 60.972, 20.787],
            [18.814, 61.911, 21.529], [19.832, 58.847, 23.477], [23.339, 59.384, 24.635],
            [24.692, 55.900, 25.407], [28.106, 55.827, 27.058], [29.826, 52.442, 27.054],
            [30.234, 50.614, 30.407], [33.564, 48.747, 30.392], [34.059, 46.556, 33.443],
            [37.469, 44.791, 33.264], [38.196, 42.240, 35.989], [41.686, 40.741, 35.714],
            [42.558, 37.819, 38.003], [46.097, 36.531, 37.608], [47.005, 33.262, 39.357],
            [50.573, 32.145, 38.746], [51.496, 28.675, 40.013], [51.451, 27.851, 43.748],
            [53.478, 24.696, 43.798], [53.065, 23.418, 47.327], [54.793, 20.087, 47.442],
            [54.139, 18.340, 50.742], [55.599, 14.854, 50.654], [54.631, 12.815, 53.657],
            [55.810, 9.248, 53.451], [54.481, 7.003, 56.196], [55.394, 3.348, 55.883],
            [53.784, 0.963, 58.348], [54.445, -2.744, 58.080], [52.527, -5.073, 60.336],
            [52.808, -8.799, 60.005], [50.566, -10.975, 62.129], [50.443, -14.667, 61.491],
            [47.878, -16.698, 63.543], [47.423, -20.370, 62.738], [44.535, -22.216, 64.603],
            [43.814, -25.822, 63.564], [40.718, -27.490, 65.210], [39.791, -30.946, 63.875],
            [36.603, -32.539, 65.287], [35.455, -35.829, 63.801], [32.205, -37.381, 65.084],
            [30.906, -40.536, 63.401], [27.658, -42.034, 64.662], [26.195, -45.054, 62.819],
            [22.890, -46.529, 63.992], [21.330, -49.372, 61.996], [17.987, -50.846, 63.097],
            [16.277, -53.521, 60.961], [12.886, -54.940, 62.027], [11.071, -57.449, 59.781],
            [7.636, -58.797, 60.780], [5.791, -61.154, 58.479], [2.296, -62.375, 59.448],
            [0.451, -64.577, 57.059], [-3.065, -65.616, 57.933], [-5.000, -67.644, 55.406],
            [-8.513, -68.508, 56.293], [-10.564, -70.406, 53.807],
        ]),
        'ss': 'CCHHHHHHHHHHHHHHHHHCCCCHHHHHHHHHHHCCCCCCHHHHHHHHHHHHHHHHHHHCCCHHHHHHHHHCCCCHHHHHHHH',
    },

    'cold_shock': {
        'name': 'Cold Shock Protein (CspB)',
        'length': 67,
        'pdb': '1CSP',
        'fold_type': 'beta',
        'constraint': 'hydrophobic',
        'folding_rate': 8.9,  # ~150 μs
        'sequence': 'MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA',
        'codons': ['ATG', 'CTG', 'GAG', 'GGC', 'AAG', 'GTG', 'AAG', 'TGG', 'TTC', 'AAC',
                   'AGC', 'GAG', 'AAG', 'GGC', 'TTC', 'GGC', 'TTC', 'ATC', 'GAG', 'GTG',
                   'GAG', 'GGC', 'CAG', 'GAC', 'GAC', 'GTG', 'TTC', 'GTG', 'CAC', 'TTC',
                   'AGC', 'GCC', 'ATC', 'CAG', 'GGC', 'GAG', 'GGC', 'TTC', 'AAG', 'ACC',
                   'CTG', 'GAG', 'GAG', 'GGC', 'CAG', 'GCC', 'GTG', 'AGC', 'TTC', 'GAG',
                   'ATC', 'GTG', 'GAG', 'GGC', 'AAC', 'CGC', 'GGC', 'CCC', 'CAG', 'GCC',
                   'GCC', 'AAC', 'GTG', 'ACC', 'AAG', 'GAG', 'GCC'],
        'ca_coords': np.array([
            [19.804, 27.696, 12.264], [18.099, 24.409, 12.761], [14.441, 23.656, 12.206],
            [12.768, 20.296, 12.462], [9.168, 19.391, 11.614], [7.436, 16.119, 11.890],
            [3.873, 15.016, 11.050], [2.211, 11.683, 11.359], [4.282, 9.437, 13.494],
            [3.091, 10.260, 17.020], [5.877, 12.525, 18.062], [8.019, 9.609, 18.782],
            [11.556, 10.717, 18.472], [13.431, 7.519, 17.846], [16.968, 8.393, 16.897],
            [18.589, 5.055, 16.299], [21.986, 5.771, 14.913], [23.438, 2.287, 14.489],
            [22.072, 0.766, 11.241], [25.054, -1.398, 10.109], [24.058, -4.931, 9.138],
            [24.629, -5.588, 5.432], [22.056, -8.210, 4.640], [22.367, -8.691, 0.876],
            [19.351, -11.058, 0.308], [18.952, -11.203, -3.454], [15.668, -13.141, -3.847],
            [14.846, -12.940, -7.545], [11.389, -14.374, -7.927], [10.190, -13.614, -11.411],
            [6.742, -14.942, -11.763], [5.145, -13.663, -14.911], [1.654, -14.852, -15.293],
            [-0.265, -12.971, -17.849], [-3.778, -13.803, -18.528], [-5.728, -11.234, -20.384],
            [-9.253, -11.835, -21.381], [-11.297, -8.772, -22.399], [-14.803, -9.098, -23.656],
            [-16.854, -5.932, -24.181], [-20.260, -6.103, -25.733], [-22.371, -2.961, -25.797],
            [-25.629, -3.147, -27.585], [-27.784, 0.004, -27.338], [-30.902, 0.128, -29.430],
            [-33.125, 3.172, -28.909], [-32.119, 5.291, -25.865], [-34.648, 7.920, -24.788],
            [-32.918, 10.033, -22.166], [-34.802, 13.213, -21.238], [-32.679, 15.192, -18.746],
            [-33.994, 18.614, -17.586], [-31.454, 20.476, -15.396], [-32.206, 23.994, -14.148],
            [-29.311, 25.668, -12.163], [-29.531, 29.195, -10.761], [-26.375, 30.637, -8.940],
            [-26.101, 34.076, -7.211], [-22.787, 35.296, -5.594], [-22.289, 38.533, -3.589],
            [-18.870, 39.513, -2.013], [-18.217, 42.482, 0.317], [-14.746, 43.307, 1.645],
            [-13.925, 45.943, 4.175], [-10.416, 46.446, 5.429], [-9.462, 48.690, 8.210],
            [-5.935, 48.956, 9.344],
        ]),
        'ss': 'CEEEEEEEECCCCCEEEEEECCCCCEEEEEECCCCCEEEEEECCCCCEEEEEECCCCCEEEEEEC',
    },

    # =========================================================================
    # SLOW FOLDERS (τ > 1 ms, ln(kf) < 7)
    # =========================================================================
    'ubiquitin': {
        'name': 'Ubiquitin',
        'length': 76,
        'pdb': '1UBQ',
        'fold_type': 'alpha/beta',
        'constraint': 'hydrophobic',
        'folding_rate': 5.5,  # ~4 ms
        'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        'codons': ['ATG', 'CAG', 'ATC', 'TTC', 'GTG', 'AAG', 'ACC', 'CTG', 'ACC', 'GGC',
                   'AAG', 'ACC', 'ATC', 'ACC', 'CTG', 'GAG', 'GTG', 'GAG', 'CCC', 'AGC',
                   'GAC', 'ACC', 'ATC', 'GAG', 'AAC', 'GTG', 'AAG', 'GCC', 'AAG', 'ATC',
                   'CAG', 'GAC', 'AAG', 'GAG', 'GGC', 'ATC', 'CCC', 'CCC', 'GAC', 'CAG',
                   'CAG', 'CGC', 'CTG', 'ATC', 'TTC', 'GCC', 'GGC', 'AAG', 'CAG', 'CTG',
                   'GAG', 'GAC', 'GGC', 'CGC', 'ACC', 'CTG', 'AGC', 'GAC', 'TAC', 'AAC',
                   'ATC', 'CAG', 'AAG', 'GAG', 'AGC', 'ACC', 'CTG', 'CAC', 'CTG', 'GTG',
                   'CTG', 'CGC', 'CTG', 'CGC', 'GGC', 'GGC'],
        'ca_coords': np.array([
            [27.340, 24.430, 2.614], [26.266, 25.413, 5.984], [23.204, 23.756, 7.149],
            [20.168, 25.896, 6.541], [17.443, 23.476, 6.238], [14.424, 23.361, 8.629],
            [13.885, 23.840, 12.410], [10.301, 24.894, 12.499], [8.058, 22.440, 14.188],
            [9.008, 22.892, 17.855], [7.664, 20.248, 20.033], [9.746, 18.134, 22.330],
            [8.473, 17.239, 25.803], [11.024, 14.532, 26.296], [10.453, 11.508, 24.095],
            [10.347, 8.144, 25.867], [8.310, 5.261, 25.055], [6.415, 5.739, 21.787],
            [8.478, 7.330, 18.944], [8.886, 10.770, 20.265], [11.495, 12.660, 18.298],
            [10.974, 16.324, 17.476], [11.411, 17.932, 13.990], [14.558, 16.096, 13.140],
            [15.275, 17.422, 9.603], [17.995, 20.024, 9.368], [16.182, 23.109, 8.473],
            [15.474, 25.165, 5.260], [16.973, 28.580, 5.458], [14.691, 30.728, 7.335],
            [14.811, 33.614, 4.926], [18.355, 33.168, 3.755], [19.063, 32.137, 7.335],
            [15.730, 33.558, 8.544], [15.568, 33.825, 12.315], [17.882, 36.753, 12.671],
            [17.113, 39.045, 15.516], [19.499, 42.015, 15.171], [22.870, 40.628, 14.136],
            [24.445, 43.387, 12.150], [28.100, 42.471, 12.526], [30.203, 45.478, 11.378],
            [27.631, 47.996, 10.298], [26.377, 48.256, 6.718], [22.814, 49.633, 6.609],
            [23.273, 51.765, 3.547], [22.675, 50.077, 0.215], [24.124, 46.602, 0.261],
            [26.853, 48.096, 2.457], [29.508, 45.524, 2.415], [32.102, 47.618, 0.614],
            [34.761, 45.095, 0.612], [33.472, 41.586, 1.401], [35.851, 39.588, 3.555],
            [33.653, 36.573, 4.323], [34.731, 35.108, 7.724], [31.641, 33.176, 8.904],
            [31.446, 33.705, 12.699], [27.842, 32.528, 13.373], [26.673, 35.259, 15.703],
            [23.259, 34.133, 17.060], [21.506, 37.436, 17.256], [18.211, 36.107, 18.655],
            [15.888, 38.795, 17.186], [12.298, 37.916, 17.930], [10.606, 40.296, 15.588],
            [7.037, 40.782, 16.636], [5.145, 43.890, 15.605], [1.406, 43.681, 15.393],
            [-0.131, 47.147, 15.477], [-3.789, 47.586, 14.563], [-5.387, 50.960, 14.642],
            [-5.119, 53.053, 17.743], [-7.532, 55.896, 17.344], [-6.026, 57.889, 20.239],
            [-8.557, 59.830, 22.230],
        ]),
        'ss': 'CEEEEEECCCCCEEEEEECCCCCCCEEEEEECCCCCCCEEEEEECCCCCHHHHHCCCEEEEEECCCHHHHHCCCCCC',
    },

    # =========================================================================
    # DISULFIDE-CONSTRAINED (for comparison)
    # =========================================================================
    'insulin_b': {
        'name': 'Insulin B-chain',
        'length': 30,
        'pdb': '4INS',
        'fold_type': 'alpha',
        'constraint': 'disulfide',
        'folding_rate': 3.0,  # Very slow due to disulfide formation
        'sequence': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
        'codons': ['TTT', 'GTG', 'AAC', 'CAG', 'CAC', 'CTG', 'TGC', 'GGC', 'AGC', 'CAC',
                   'CTG', 'GTG', 'GAG', 'GCC', 'CTG', 'TAC', 'CTG', 'GTG', 'TGC', 'GGC',
                   'GAG', 'CGC', 'GGC', 'TTC', 'TTC', 'TAC', 'ACC', 'CCC', 'AAG', 'ACC'],
        'ca_coords': np.array([
            [12.86, 3.91, 5.44], [10.41, 1.04, 5.87], [11.06, -2.54, 4.74],
            [8.15, -4.90, 4.73], [8.09, -6.97, 1.63], [5.30, -9.45, 1.13],
            [5.93, -11.16, -2.17], [3.77, -14.24, -2.23], [5.11, -17.64, -1.28],
            [3.62, -18.50, 1.97], [4.18, -15.02, 3.36], [2.57, -13.82, 6.53],
            [3.97, -10.33, 7.18], [1.66, -8.64, 9.76], [2.21, -4.92, 10.21],
            [-0.25, -3.05, 12.22], [-0.45, 0.76, 11.53], [-2.61, 2.40, 14.04],
            [-2.01, 6.14, 14.10], [-4.33, 8.63, 12.24], [-3.21, 11.88, 10.71],
            [-5.07, 13.41, 7.87], [-3.04, 16.59, 7.33], [-4.27, 18.21, 4.16],
            [-1.82, 21.00, 3.59], [-2.52, 22.65, 0.24], [0.67, 24.59, -0.68],
            [1.15, 26.10, -4.07], [4.59, 27.54, -4.82], [5.08, 29.17, -8.07],
        ]),
        'ss': 'CCHHHHHHHCCHHHHHHHHHCCCHHHHCCCC',
    },

    'bpti': {
        'name': 'BPTI',
        'length': 58,
        'pdb': '5PTI',
        'fold_type': 'alpha/beta',
        'constraint': 'disulfide',
        'folding_rate': 2.0,  # Very slow, 3 disulfides
        'sequence': 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
        'codons': ['CGC', 'CCC', 'GAC', 'TTC', 'TGC', 'CTG', 'GAG', 'CCC', 'CCC', 'TAC',
                   'ACC', 'GGC', 'CCC', 'TGC', 'AAG', 'GCC', 'CGC', 'ATC', 'ATC', 'CGC',
                   'TAC', 'TTC', 'TAC', 'AAC', 'GCC', 'AAG', 'GCC', 'GGC', 'CTG', 'TGC',
                   'CAG', 'ACC', 'TTC', 'GTG', 'TAC', 'GGC', 'GGC', 'TGC', 'CGC', 'GCC',
                   'AAG', 'CGC', 'AAC', 'AAC', 'TTC', 'AAG', 'AGC', 'GCC', 'GAG', 'GAC',
                   'TGC', 'ATG', 'CGC', 'ACC', 'TGC', 'GGC', 'GGC', 'GCC'],
        'ca_coords': np.array([
            [32.800, 30.192, 12.879], [31.797, 28.357, 16.029], [34.073, 25.404, 16.224],
            [32.720, 23.252, 19.077], [34.970, 20.146, 19.093], [32.973, 16.944, 18.746],
            [30.553, 16.680, 21.670], [32.780, 15.018, 24.195], [31.046, 13.045, 26.872],
            [30.987, 9.244, 26.710], [28.697, 7.706, 29.327], [29.629, 4.039, 29.594],
            [28.277, 2.291, 32.678], [30.892, 0.052, 34.204], [34.268, 1.755, 34.137],
            [35.498, 4.951, 32.545], [38.923, 5.141, 34.186], [40.587, 8.417, 33.438],
            [38.839, 11.614, 32.547], [38.095, 13.002, 29.102], [38.972, 16.700, 28.596],
            [36.688, 19.296, 27.096], [35.917, 22.688, 28.461], [32.679, 24.353, 27.296],
            [30.754, 24.571, 24.075], [28.266, 21.727, 23.716], [30.188, 19.029, 25.460],
            [32.832, 17.269, 23.395], [31.553, 15.296, 20.356], [29.100, 13.102, 21.959],
            [30.825, 9.676, 22.152], [30.295, 7.932, 18.717], [32.609, 5.001, 18.429],
            [30.847, 1.831, 17.527], [27.196, 2.181, 16.675], [25.523, 1.605, 13.299],
            [21.828, 1.026, 13.831], [20.098, 4.281, 14.731], [18.413, 5.312, 11.425],
            [14.679, 5.725, 11.707], [13.045, 8.988, 11.034], [13.948, 11.066, 8.166],
            [11.193, 13.472, 7.435], [12.430, 14.800, 4.035], [9.829, 17.443, 3.604],
            [10.944, 19.063, 0.303], [8.395, 21.823, -0.144], [9.548, 23.692, -3.268],
            [7.064, 26.560, -3.443], [8.168, 28.595, -6.334], [5.655, 31.479, -6.287],
            [6.573, 33.709, -9.157], [4.178, 36.607, -9.180], [5.013, 38.944, -12.060],
            [2.748, 41.891, -12.091], [3.435, 44.394, -14.898], [0.915, 47.279, -14.900],
            [1.414, 50.132, -17.395],
        ]),
        'ss': 'CCCCCCCCCCCCCCCCHHHHHHHHHCCCCCCCEEEECCCCCCCCCCCCCCCCCCCCCCCC',
    },

    'crambin': {
        'name': 'Crambin',
        'length': 46,
        'pdb': '1CRN',
        'fold_type': 'alpha/beta',
        'constraint': 'disulfide',
        'folding_rate': 2.5,  # Slow, 3 disulfides
        'sequence': 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
        'codons': ['ACC', 'ACC', 'TGC', 'TGC', 'CCC', 'AGC', 'ATC', 'GTG', 'GCC', 'CGC',
                   'AGC', 'AAC', 'TTC', 'AAC', 'GTG', 'TGC', 'CGC', 'CTG', 'CCC', 'GGC',
                   'ACC', 'CCC', 'GAG', 'GCC', 'ATC', 'TGC', 'GCC', 'ACC', 'TAC', 'ACC',
                   'GGC', 'TGC', 'ATC', 'ATC', 'ATC', 'CCC', 'GGC', 'GCC', 'ACC', 'TGC',
                   'CCC', 'GGC', 'GAC', 'TAC', 'GCC', 'AAC'],
        'ca_coords': np.array([
            [17.047, 14.099, 3.625], [16.967, 12.784, 6.469], [15.685, 9.617, 5.285],
            [13.390, 10.272, 2.473], [12.631, 7.344, 0.396], [16.134, 6.071, 0.443],
            [17.225, 4.478, 3.678], [14.329, 4.083, 5.870], [12.070, 1.466, 5.457],
            [13.193, -1.226, 7.572], [10.488, -3.733, 7.574], [7.171, -2.175, 8.479],
            [5.678, -5.199, 6.886], [6.431, -5.170, 3.088], [9.714, -6.791, 3.336],
            [11.778, -3.827, 2.644], [10.436, -0.391, 2.099], [7.629, -0.858, -0.521],
            [7.904, 1.896, -2.791], [10.633, 4.118, -2.128], [9.335, 7.639, -2.614],
            [12.245, 9.798, -1.478], [13.603, 12.428, -4.070], [16.866, 10.911, -4.851],
            [17.065, 11.160, -8.667], [14.062, 12.959, -10.034], [13.476, 10.748, -13.078],
            [9.727, 10.811, -13.167], [8.973, 9.453, -16.579], [5.296, 9.867, -17.331],
            [4.485, 6.334, -18.471], [4.986, 3.443, -16.057], [8.551, 4.013, -15.119],
            [9.929, 1.156, -17.103], [13.621, 1.510, -17.695], [14.879, 2.399, -14.195],
            [18.453, 1.357, -14.776], [18.593, -1.846, -13.017], [21.619, -2.035, -10.731],
            [20.127, -2.863, -7.409], [22.647, -3.630, -4.580], [20.853, -3.875, -1.250],
            [22.699, -1.574, 1.196], [20.075, -0.430, 3.750], [21.386, 2.900, 5.060],
            [18.043, 4.589, 5.318],
        ]),
        'ss': 'CCCCCCEEECCCCCCCCCCCCCCCCCCCHHHHHCEEEEECCCCCCCC',
    },
}

# Amino acid properties for hydrophobic contact analysis
HYDROPHOBIC_AA = set('AILMFVPGW')
POLAR_AA = set('STYCNQ')
CHARGED_AA = set('DEKRH')


def load_codon_mapping():
    """Load codon to embedding position mapping."""
    map_path = Path(__file__).parent.parent / 'embeddings' / 'codon_mapping_3adic.json'
    with open(map_path) as f:
        return json.load(f)['codon_to_position']


def load_embeddings():
    """Load pre-extracted embeddings."""
    emb_path = Path(__file__).parent.parent / 'embeddings' / 'v5_11_3_embeddings.pt'
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    return emb_data['z_B_hyp']


def compute_contact_map(coords, threshold=8.0, min_seq_sep=4):
    """Compute binary contact map from Cα coordinates."""
    n = len(coords)
    dist_matrix = squareform(pdist(coords))
    contact_map = (dist_matrix < threshold).astype(float)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < min_seq_sep:
                contact_map[i, j] = 0
    return contact_map, dist_matrix


def evaluate_protein_detailed(protein_id, z_hyp, codon_to_pos):
    """Detailed evaluation with multiple metrics."""
    protein = PROTEINS[protein_id]
    codons = protein['codons']
    coords = protein['ca_coords']
    sequence = protein['sequence']

    n = min(len(codons), len(coords), len(sequence))
    codons = codons[:n]
    coords = coords[:n]
    sequence = sequence[:n]

    if n < 12:
        return None

    contact_map, dist_matrix = compute_contact_map(coords)
    n_contacts = int(contact_map.sum() / 2)

    if n_contacts < 3:
        return None

    # Collect all pairs with metadata
    pairs_data = []

    for i in range(n):
        for j in range(i + 4, n):
            if codons[i] not in codon_to_pos or codons[j] not in codon_to_pos:
                continue

            idx_i = codon_to_pos[codons[i]]
            idx_j = codon_to_pos[codons[j]]

            hyp_dist = poincare_distance(
                z_hyp[idx_i:idx_i+1],
                z_hyp[idx_j:idx_j+1],
                c=1.0
            ).item()

            # Determine contact type
            aa_i, aa_j = sequence[i], sequence[j]
            if aa_i in HYDROPHOBIC_AA and aa_j in HYDROPHOBIC_AA:
                contact_type = 'hydrophobic'
            elif aa_i in CHARGED_AA or aa_j in CHARGED_AA:
                contact_type = 'charged'
            else:
                contact_type = 'polar'

            pairs_data.append({
                'i': i, 'j': j,
                'seq_sep': j - i,
                'hyp_dist': hyp_dist,
                'contact': contact_map[i, j],
                'ca_dist': dist_matrix[i, j],
                'contact_type': contact_type,
            })

    if not pairs_data:
        return None

    # Convert to arrays
    hyp_dists = np.array([p['hyp_dist'] for p in pairs_data])
    contacts = np.array([p['contact'] for p in pairs_data])
    seq_seps = np.array([p['seq_sep'] for p in pairs_data])
    contact_types = [p['contact_type'] for p in pairs_data]

    # Overall AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc_overall = roc_auc_score(contacts, -hyp_dists)
    except ValueError:
        auc_overall = 0.5

    # AUC by sequence separation range
    auc_by_range = {}
    for range_name, (min_sep, max_sep) in [
        ('local', (4, 8)),
        ('medium', (8, 16)),
        ('long', (16, 100)),
    ]:
        mask = (seq_seps >= min_sep) & (seq_seps < max_sep)
        if mask.sum() > 10 and contacts[mask].sum() > 0:
            try:
                auc_by_range[range_name] = roc_auc_score(contacts[mask], -hyp_dists[mask])
            except:
                auc_by_range[range_name] = 0.5

    # AUC by contact type
    auc_by_type = {}
    for ctype in ['hydrophobic', 'charged', 'polar']:
        mask = np.array([ct == ctype for ct in contact_types])
        if mask.sum() > 10 and contacts[mask].sum() > 0:
            try:
                auc_by_type[ctype] = roc_auc_score(contacts[mask], -hyp_dists[mask])
            except:
                auc_by_type[ctype] = 0.5

    # Cohen's d
    contact_dists = hyp_dists[contacts == 1]
    noncontact_dists = hyp_dists[contacts == 0]
    if len(contact_dists) > 0 and len(noncontact_dists) > 0:
        pooled_std = np.sqrt((contact_dists.var() + noncontact_dists.var()) / 2)
        cohens_d = (contact_dists.mean() - noncontact_dists.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0

    return {
        'id': protein_id,
        'name': protein['name'],
        'length': n,
        'fold_type': protein['fold_type'],
        'constraint': protein['constraint'],
        'folding_rate': protein.get('folding_rate', None),
        'n_contacts': n_contacts,
        'n_pairs': len(pairs_data),
        'auc_overall': auc_overall,
        'cohens_d': cohens_d,
        'auc_by_range': auc_by_range,
        'auc_by_type': auc_by_type,
    }


def main():
    print("=" * 90)
    print("DEEP ANALYSIS: Multiple Hypotheses Testing")
    print("=" * 90)
    print()

    # Load embeddings
    print("Loading embeddings...")
    z_hyp = load_embeddings()
    codon_to_pos = load_codon_mapping()
    print(f"  Testing {len(PROTEINS)} proteins")
    print()

    # Evaluate all proteins
    results = []
    for pid in sorted(PROTEINS.keys(), key=lambda x: PROTEINS[x].get('folding_rate', 0), reverse=True):
        result = evaluate_protein_detailed(pid, z_hyp, codon_to_pos)
        if result:
            results.append(result)
            rate = result['folding_rate']
            rate_str = f"ln(kf)={rate:.1f}" if rate else "N/A"
            print(f"  {result['name']:<25} AUC={result['auc_overall']:.3f}  d={result['cohens_d']:+.3f}  {rate_str}")

    # =========================================================================
    # ANALYSIS 1: Folding Rate Correlation
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 1: Folding Rate vs Contact Prediction Signal")
    print("=" * 90)

    proteins_with_rates = [r for r in results if r['folding_rate'] is not None]
    if len(proteins_with_rates) >= 5:
        rates = np.array([r['folding_rate'] for r in proteins_with_rates])
        aucs = np.array([r['auc_overall'] for r in proteins_with_rates])

        r_rate, p_rate = stats.spearmanr(rates, aucs)
        print(f"\n  Spearman correlation (ln(kf) vs AUC): ρ = {r_rate:.4f} (p = {p_rate:.4f})")

        if r_rate > 0.3 and p_rate < 0.1:
            print("  >>> CONFIRMED: Faster folding proteins encode physics better!")
        print()
        print("  By folding speed category:")

        for category, (min_rate, max_rate) in [
            ('Ultrafast (τ < 10 μs)', (11.5, 20)),
            ('Fast (τ ~ 100 μs)', (7.5, 11.5)),
            ('Slow (τ > 1 ms)', (0, 7.5)),
        ]:
            cat_results = [r for r in proteins_with_rates if min_rate <= r['folding_rate'] < max_rate]
            if cat_results:
                mean_auc = np.mean([r['auc_overall'] for r in cat_results])
                print(f"    {category}: n={len(cat_results)}, mean AUC = {mean_auc:.3f}")

    # =========================================================================
    # ANALYSIS 2: Contact Range Analysis
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 2: Local vs Long-Range Contacts")
    print("=" * 90)

    range_labels = {'local': '4-8', 'medium': '8-16', 'long': '>16'}
    for range_name in ['local', 'medium', 'long']:
        range_aucs = [r['auc_by_range'].get(range_name) for r in results if range_name in r['auc_by_range']]
        if range_aucs:
            print(f"\n  {range_name.upper()} contacts (|i-j| = {range_labels[range_name]}):")
            print(f"    Mean AUC = {np.mean(range_aucs):.4f} ± {np.std(range_aucs):.4f}")

    # =========================================================================
    # ANALYSIS 3: Contact Type Analysis
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 3: Hydrophobic vs Polar vs Charged Contacts")
    print("=" * 90)

    for ctype in ['hydrophobic', 'charged', 'polar']:
        type_aucs = [r['auc_by_type'].get(ctype) for r in results if ctype in r['auc_by_type']]
        if type_aucs:
            print(f"\n  {ctype.upper()} contacts:")
            print(f"    Mean AUC = {np.mean(type_aucs):.4f} ± {np.std(type_aucs):.4f} (n={len(type_aucs)})")

    # =========================================================================
    # ANALYSIS 4: By Constraint Type
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 4: By Constraint Type")
    print("=" * 90)

    for constraint in ['hydrophobic', 'designed', 'disulfide']:
        const_results = [r for r in results if r['constraint'] == constraint]
        if const_results:
            mean_auc = np.mean([r['auc_overall'] for r in const_results])
            print(f"\n  {constraint.upper()}: n={len(const_results)}, mean AUC = {mean_auc:.4f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    overall_aucs = [r['auc_overall'] for r in results]
    t_stat, p_val = stats.ttest_1samp(overall_aucs, 0.5)

    print(f"\n  Overall: Mean AUC = {np.mean(overall_aucs):.4f} ± {np.std(overall_aucs):.4f}")
    print(f"  t-test (AUC > 0.5): t = {t_stat:.4f}, p = {p_val:.6f}")

    if p_val < 0.01:
        print("\n  >>> HIGHLY SIGNIFICANT (p < 0.01)")
    elif p_val < 0.05:
        print("\n  >>> SIGNIFICANT (p < 0.05)")

    # Save detailed results
    output_file = Path(__file__).parent.parent / 'data' / 'deep_analysis_results.json'
    output_file.parent.mkdir(exist_ok=True)

    # Convert to serializable format
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        serializable_results.append(sr)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return results


if __name__ == '__main__':
    results = main()
