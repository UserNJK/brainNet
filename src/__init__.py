"""
BioNeuron — a biologically accurate spiking neural network library.

Public API:

    from bioneuron import BrainNet, NeuronParams
    from bioneuron import default_neurochemistry, ClosedLoop

Core simulation (`core`), the sparse connectivity backend (`sparse_synapses`),
the neurochemical system (`neurochemistry`) and the closed-loop learning harness
(`closed_loop`) are all re-exported here. See the README for the full guide.
"""

__version__ = "1.8.0"
__author__ = "Jugal Kishore"

from .core import (
    BrainNet,
    NeuronParams,
    NeuromodulatorState,
    SynapseMatrix,
    NeuronPopulation,
    GrowthCone,
    ThoughtDetector,
    OscillationAnalyzer,
    CriticalityAnalyzer,
    SpikeStatistics,
)
from .sparse_synapses import SparseSynapses
from .neurochemistry import (
    Neurochemical,
    Neurochemistry,
    default_neurochemistry,
    stressed,
    drowsy,
    aroused,
)
from .closed_loop import (
    ClosedLoop,
    accuracy_curve,
    demo_conditioning,
    demo_discrimination,
)
from .cortex import (
    CellType,
    RS, FS, LTS,
    build_cortex,
    demo_gamma,
    excitatory_indices,
    fast_spiking_indices,
)

__all__ = [
    "__version__",
    # core
    "BrainNet", "NeuronParams", "NeuromodulatorState", "SynapseMatrix",
    "NeuronPopulation", "GrowthCone", "ThoughtDetector",
    "OscillationAnalyzer", "CriticalityAnalyzer", "SpikeStatistics",
    # backends
    "SparseSynapses",
    # neurochemistry
    "Neurochemical", "Neurochemistry", "default_neurochemistry",
    "stressed", "drowsy", "aroused",
    # closed loop
    "ClosedLoop", "accuracy_curve", "demo_conditioning", "demo_discrimination",
    # structured cortex
    "CellType", "RS", "FS", "LTS", "build_cortex", "demo_gamma",
    "excitatory_indices", "fast_spiking_indices",
]
