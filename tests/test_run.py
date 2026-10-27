"""End-to-end run test. Verifies that state actually flows
caspule -> detector -> caspule (atom removal) and
caspule -> detector -> nfsim (species injection)."""

import importlib
import os

import pytest
from process_bigraph import Composite, gather_emitter_results

lammps_available = importlib.util.find_spec("lammps") is not None
needs_lammps = pytest.mark.skipif(
    not lammps_available,
    reason="lammps Python bindings not installed (not available on arm64 mac via pip)",
)

from pbg_composite_nfsim_caspule.core import build_core
from pbg_composite_nfsim_caspule.document import build_document


HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.normpath(os.path.join(HERE, '..', 'examples'))


# Slightly bigger box than test_assembly's so a few bonds can form
# during the run — the detector then has something to detect.
INLINE_CASPULE = """
units lj
atom_style bond
boundary p p p
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
region box block 0 8 0 8 0 8
create_box 1 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 50
mass 1 1.0
lattice sc 1.6
create_atoms 1 box
pair_style soft 2.0
pair_coeff * * 30.0
bond_style harmonic
bond_coeff 1 5.0 1.5
minimize 1e-4 1e-4 100 100
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


def _make_sim(decoupled=False):
    core = build_core()
    doc = build_document(
        caspule_input_script=INLINE_CASPULE,
        nfsim_model_file=os.path.join(EXAMPLES, 'nfsim_dimer_kinetics.bngl'),
        detector_rules=INLINE_RULES,
        caspule_interval=0.05,
        nfsim_interval=0.5,
        nfsim_n_steps=20,
        decoupled=decoupled,
    )
    return Composite({'state': doc}, core=core)


@needs_lammps
@pytest.mark.timeout(120)
def test_dimer_flows_caspule_to_nfsim():
    """Run long enough for at least one bond to form, then assert that
    the species store has a non-zero Dimer count *and* that CASPULE
    lost atoms — the only way both are true is if the full chain
    (caspule out -> detector -> nfsim in / caspule in) ran."""
    sim = _make_sim(decoupled=False)
    initial_atoms = sim.state['caspule']['instance']._lmp.get_natoms() \
        if hasattr(sim.state['caspule'], 'instance') else None
    sim.run(1.0)

    species = sim.state['stores']['species']
    # Detector should have credited at least one Dimer.
    assert species.get('Dimer', 0.0) + species.get('Active', 0.0) > 0, (
        f'no species injected — detector or nfsim wiring broken; '
        f'species={species}')

    # And atoms_to_remove should have been delivered: the spatial pool
    # shrinks vs the lattice population (the dimerize.in lattice fills
    # the box with type-1 atoms; the exact count depends on LAMMPS's
    # `lattice sc 1.6` interpretation but is well over 100).
    num_atoms = sim.state['stores']['num_atoms']
    assert num_atoms < 100, (
        f'CASPULE never lost atoms — atoms_to_remove store not consumed; '
        f'num_atoms={num_atoms}')


@needs_lammps
@pytest.mark.timeout(120)
def test_decoupled_baseline_keeps_atoms_and_species_zero():
    """Sanity check: with decoupled=True the detector still runs
    (against the same CASPULE) but its outputs go nowhere, so:
        - CASPULE keeps all atoms
        - NFSim's species pool stays empty
    If this test fails it means the wiring used by the coupled run
    is leaking into the baseline (or vice versa)."""
    sim = _make_sim(decoupled=True)
    sim.run(1.0)

    species = sim.state['stores']['species']
    assert sum(species.values()) == 0.0, (
        f'decoupled run injected species — wiring leak: {species}')

    num_atoms = sim.state['stores']['num_atoms']
    assert num_atoms == 1000, (
        f'decoupled run lost atoms — atoms_to_remove leaking through; '
        f'num_atoms={num_atoms}')
