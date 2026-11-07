import time
from pathlib import Path

from mitraav.core.finding import Finding
from mitraav.core.probe import Probe
from mitraav.core.runner import run_probe


class Fine(Probe):
    name = "fine"
    def run(self, path):
        return Finding(probe=self.name, detected=True, strength=1, narrows=1)


class Boom(Probe):
    name = "boom"
    def run(self, path):
        raise ValueError("nope")


class Sleepy(Probe):
    name = "sleepy"
    timeout = 1.0
    def run(self, path):
        time.sleep(30)
        return Finding(probe=self.name, detected=True)


def test_ok():
    assert run_probe(Fine, Path(".")).detected


def test_exception_becomes_error():
    f = run_probe(Boom, Path("."))
    assert f.status == "error" and "nope" in f.detail


def test_timeout_is_killed():
    start = time.time()
    assert run_probe(Sleepy, Path(".")).status == "timeout"
    assert time.time() - start < 10
