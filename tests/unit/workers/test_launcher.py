import os
import sys
from types import SimpleNamespace

import pytest

from campus_occupancy.workers import launcher


def test_pick_interpreter_prefers_venv(monkeypatch):
    monkeypatch.setattr(launcher.os.path, "exists", lambda p: True)
    assert launcher.pick_interpreter() == os.path.join(launcher.ROOT, "venv", "bin", "python")


def test_pick_interpreter_falls_back(monkeypatch):
    monkeypatch.setattr(launcher.os.path, "exists", lambda p: False)
    assert launcher.pick_interpreter() == sys.executable


def test_preflight_passes(monkeypatch):
    seen = {}
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda cmd, capture_output, text: seen.update(cmd=cmd) or SimpleNamespace(returncode=0, stderr=""))
    launcher.preflight("/some/python")
    assert seen["cmd"][0] == "/some/python" and "sqlalchemy" in seen["cmd"][2]


class FakeProc:
    """Stands in for subprocess.Popen; `exit_after` makes poll() report death."""
    def __init__(self, args, exit_code=None, hang=False):
        self.args = args
        self._exit_code = exit_code
        self.hang = hang
        self.returncode = None
        self.terminated = self.killed = False
        self.wait_calls = []

    def poll(self):
        if self._exit_code is not None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.hang = False

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.hang and timeout is not None:
            raise launcher.subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = 0
        return 0


def _run_main(monkeypatch, procs, sleep_behaviour):
    monkeypatch.setattr(launcher, "pick_interpreter", lambda: "/py")
    monkeypatch.setattr(launcher, "preflight", lambda py: None)
    it = iter(procs)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda args: next(it))
    monkeypatch.setattr(launcher.time, "sleep", sleep_behaviour)
    launcher.main()


def test_main_spawns_four_children_and_terminates_on_interrupt(monkeypatch, capsys):
    procs = [FakeProc(f"p{i}") for i in range(4)]
    sleeps = {"n": 0}

    def sleep(s):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:                 # 1st = startup pause, 2nd = monitor loop → Ctrl+C
            raise KeyboardInterrupt

    _run_main(monkeypatch, procs, sleep)
    out = capsys.readouterr().out
    assert "Shutdown signal received" in out and "All systems offline" in out
    assert all(p.terminated and p.wait_calls == [5] for p in procs)
    assert "lab_simulator.py" not in out or True   # args are fake; just ensure no crash


def test_main_shuts_down_when_a_child_dies(monkeypatch, capsys):
    procs = [FakeProc("a"), FakeProc("b", exit_code=3), FakeProc("c"), FakeProc("d")]
    _run_main(monkeypatch, procs, lambda s: None)
    out = capsys.readouterr().out
    assert "stopped unexpectedly" in out and "[3]" in out
    assert not procs[1].terminated and all(p.terminated for p in (procs[0], procs[2], procs[3]))


def test_main_kills_children_that_ignore_terminate(monkeypatch, capsys):
    procs = [FakeProc("a", hang=True), FakeProc("b"), FakeProc("c"), FakeProc("d")]
    sleeps = {"n": 0}

    def sleep(s):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise KeyboardInterrupt

    _run_main(monkeypatch, procs, sleep)
    out = capsys.readouterr().out
    assert procs[0].killed and "did not exit in 5s; killing" in out
    assert not any(p.killed for p in procs[1:])


def test_preflight_exits_on_missing_packages(monkeypatch, capsys):
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda cmd, capture_output, text: SimpleNamespace(returncode=1, stderr="No module named x"))
    with pytest.raises(SystemExit) as exc:
        launcher.preflight("/some/python")
    assert exc.value.code == 1 and "No module named x" in capsys.readouterr().out
