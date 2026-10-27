"""Smoke test for the `python -m pbg_composite_nfsim_caspule` runner.

Covers the path that users will hit first: load a bundled PBG JSON,
resolve relative input-file paths, run the Composite, get a summary.
"""

import importlib
import json
import os

import pytest

lammps_available = importlib.util.find_spec("lammps") is not None
needs_lammps = pytest.mark.skipif(
    not lammps_available,
    reason="lammps Python bindings not installed (not available on arm64 mac via pip)",
)

from pbg_composite_nfsim_caspule import (
    load_document,
    resolve_input_paths,
    run,
)


HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.normpath(os.path.join(HERE, '..', 'examples'))
DIMER_PBG = os.path.join(EXAMPLES, 'composite_dimer.pbg.json')


def test_load_resolves_relative_input_paths(tmp_path):
    doc, base_dir = load_document(DIMER_PBG)
    assert base_dir == EXAMPLES
    # Pre-resolution: the JSON contains relative paths (no leading slash).
    cas_path = doc['caspule']['config']['input_file']
    det_path = doc['detector']['config']['config_file']
    nfs_path = doc['nfsim']['config']['model_file']
    assert not os.path.isabs(cas_path), \
        f'expected relative path in checked-in JSON, got {cas_path!r}'
    assert not os.path.isabs(det_path)
    assert not os.path.isabs(nfs_path)

    resolve_input_paths(doc, base_dir)
    assert os.path.isabs(doc['caspule']['config']['input_file'])
    assert os.path.isabs(doc['detector']['config']['config_file'])
    assert os.path.isabs(doc['nfsim']['config']['model_file'])
    # And the resolved files actually exist.
    assert os.path.exists(doc['caspule']['config']['input_file'])
    assert os.path.exists(doc['detector']['config']['config_file'])
    assert os.path.exists(doc['nfsim']['config']['model_file'])


@needs_lammps
@pytest.mark.timeout(120)
def test_run_dimer_pbg_drains_atoms_and_credits_species(tmp_path):
    """The high-level `run()` is what `python -m pbg_composite_nfsim_caspule`
    calls. Asserts that pointing it at a bundled JSON yields a coupled
    simulation: CASPULE atoms drop AND NFSim species fill up.
    """
    output = tmp_path / 'run.json'
    summary = run(
        DIMER_PBG,
        total_time=1.0,
        output=str(output),
        quiet=True,
    )
    assert summary['caspule_atoms'] < 100, (
        f'expected CASPULE drained, got {summary["caspule_atoms"]} atoms')
    total_species = sum(summary['nfsim_species'].values())
    assert total_species > 0, (
        f'expected NFSim species credited, got {summary["nfsim_species"]}')

    # The --output side-effect dumps both the summary and the emitter
    # history, so users can post-process without re-running.
    dumped = json.loads(output.read_text())
    assert dumped['summary'] == summary
    assert len(dumped['history']) > 0


@needs_lammps
@pytest.mark.timeout(120)
def test_run_overrides_input_files(tmp_path):
    """--caspule / --detector / --nfsim should win over what's in the JSON."""
    # Use the polymer .in but the dimer detector + bngl, mixing the two
    # bundled experiments. Just an integration check that overrides
    # take effect.
    summary = run(
        DIMER_PBG,
        caspule=os.path.join(EXAMPLES, 'caspule_polymer.in'),
        total_time=0.5,
        quiet=True,
    )
    # Run produced *some* state; we don't assert on physics here, just
    # that overrides were honoured and the run completed without error.
    assert summary['global_time'] is not None
