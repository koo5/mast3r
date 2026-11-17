"""sparse_ga's RESTORE (hillview: a dense arm reuses a finished run's solve).

sparse_scene_optimizer(restore=...) runs its setup on the run's own inputs, checks a finished run's final state
against them (the same canonical views in order, the same anchor grid, the same anchor counts) and puts it where the
two stages would have left theirs. The optimiser cannot run alone here: the end-to-end check -- a restored solve
reproduces its source's every product bit for bit, a dense parameter changes only the dense cloud, another pairing
is refused before the optimiser -- is hillview's scripts/enrich/test_solve_from_run.py, on real solves. Checked
here: a solve without `restore` is the solve it was, and the caller can tell a checkout that has it."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga  # noqa: E402


def test_the_optimiser_optimises_by_default():
    assert inspect.signature(sparse_ga.sparse_scene_optimizer).parameters['restore'].default is None
    assert sparse_ga.RESTORE is True
