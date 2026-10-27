"""Build the document and instantiate the Composite. Catches schema-
reconciliation failures and missing wiring before any process actually
runs."""

import importlib
import os

import pytest
from process_bigraph import Composite

from pbg_composite_nfsim_caspule.core import build_core
from pbg_composite_nfsim_caspule.document import build_document

lammps_available = importlib.util.find_spec("lammps") is not None
needs_lammps = pytest.mark.skipif(
    not lammps_available,
    reason="lammps Python bindings not installed (not available on arm64 mac via pip)",
)


HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.normpath(os.path.join(HERE, '..', 'examples'))


# A small inline LAMMPS script — keeps the test fully offline and
# fast (a few atoms is plenty to exercise the wiring).
INLINE_CASPULE = """
units lj
atom_style bond
boundary p p p
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
region box block 0 6 0 6 0 6
create_box 1 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 50
mass 1 1.0
lattice sc 1.6
create_atoms 1 box
pair_style soft 2.0
pair_coeff * * 30.0
bond_style harmonic
bond_coeff 1 5.0 1.5
minimize 1e-4 1e-4 50 50
fix nve all nve/limit 0.05
fix lan all langevin 0.4 0.4 1.0 12345
timestep 0.005
fix mkbnd all bond/create 1 1 1 1.8 1 iparam 1 1 jparam 1 1
thermo 5
"""

INLINE_RULES = [
    {'name': 'Dimer', 'bond_type_cluster': 1,
     'min_size': 2, 'max_size': 2, 'remove_from_caspule': True},
]


def _bngl_path():
    return os.path.join(EXAMPLES, 'nfsim_dimer_kinetics.bngl')


def test_document_builds_without_errors():
    """build_document() returns a dict with the three processes
    plus stores plus emitter, all wired through stable paths."""
    doc = build_document(
        caspule_input_script=INLINE_CASPULE,
        nfsim_model_file=_bngl_path(),
        detector_rules=INLINE_RULES,
    )
    assert 'caspule' in doc
    assert 'detector' in doc
    assert 'nfsim' in doc
    assert 'stores' in doc
    assert 'emitter' in doc

    # The atoms_to_remove store sits between detector and caspule on
    # the same path — both must point to it.
    det_path = doc['detector']['outputs']['atoms_to_remove']
    cas_path = doc['caspule']['inputs']['atoms_to_remove']
    assert det_path == cas_path

    # The species store is shared by detector additions and nfsim I/O.
    add_path = doc['detector']['outputs']['molecule_additions']
    nfs_in   = doc['nfsim']['inputs']['observables']
    nfs_out  = doc['nfsim']['outputs']['observables']
    assert add_path == nfs_in == nfs_out


@needs_lammps
def test_composite_instantiates():
    """Composite() raises if schemas don't reconcile across stores."""
    core = build_core()
    doc = build_document(
        caspule_input_script=INLINE_CASPULE,
        nfsim_model_file=_bngl_path(),
        detector_rules=INLINE_RULES,
    )
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


@needs_lammps
def test_decoupled_baseline_also_assembles():
    """The baseline must build too — detector outputs route to a
    dead-end namespace but the schemas still have to line up."""
    core = build_core()
    doc = build_document(
        caspule_input_script=INLINE_CASPULE,
        nfsim_model_file=_bngl_path(),
        detector_rules=INLINE_RULES,
        decoupled=True,
    )
    sim = Composite({'state': doc}, core=core)
    assert sim is not None
