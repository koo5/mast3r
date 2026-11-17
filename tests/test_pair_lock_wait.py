"""A WAIT THAT SAYS SO: a run blocked on another run's pair lock says every LOCK_WAIT_SAY_S whom it waits
on -- the holder's pid and run, from /proc/locks -- so the worker's idle kill never takes an honest wait for a hang."""
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402


def test_a_blocking_wait_names_the_run_that_holds_the_pair(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sga, 'LOCK_WAIT_SAY_S', 0.2)
    monkeypatch.setattr(sga, 'LOCK_POLL_S', 0.02)
    path = str(tmp_path / '.locks' / 'a-b.lock')
    os.makedirs(os.path.dirname(path))
    holder = subprocess.Popen(
        [sys.executable, '-c', 'import fcntl, sys, time\n'
         'f = open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX); print("held", flush=True); time.sleep(30)',
         path, '--job', '/runs/abc123-w16-1/job.json'], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'held'
        lock, got = sga._PairLock(path), []
        waiter = threading.Thread(target=lambda: got.append(lock.acquire(True)))
        waiter.start()
        time.sleep(0.7)
        assert not got                                            # still waiting...
        said = capsys.readouterr().out
        assert 'waiting' in said and f'held by pid {holder.pid} (abc123-w16-1)' in said, said
    finally:
        holder.kill()
        holder.wait()
    waiter.join(5)
    assert got == [True] and lock.waited >= 0.6                  # ...and taken once the holder is gone
    lock.__exit__(None, None, None)


def test_a_tried_lock_returns_at_once_and_says_nothing(tmp_path, capsys):
    path = str(tmp_path / 'a-b.lock')
    with open(path, 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)                         # another open file: another holder, this process
        t0 = time.time()
        assert sga._PairLock(path).acquire(False) is False
        assert time.time() - t0 < 0.5 and capsys.readouterr().out == ''
        assert sga._lock_holder(path) == f'pid {os.getpid()} (not a reconstruct.py --job)'
    assert sga._lock_holder(path) is None                        # released: nobody listed
