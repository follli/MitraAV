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


class Publisher(Probe):
    name = "publisher"
    order = 1
    def run(self, path):
        return Finding(probe=self.name, detected=False, meta={"context": {"scene": {"kind": "street"}}})


class Reader(Probe):
    name = "reader"
    order = 2
    def run(self, path):
        return Finding(probe=self.name, detected=False, summary=str(self.context.get("scene", {}).get("kind")))


def test_context_published_by_one_probe_reaches_the_next(monkeypatch):
    from mitraav.core import runner
    monkeypatch.setattr(runner, "planned", lambda path: [(Publisher, "direct"), (Reader, "direct")])
    out = {f.probe: f for f in runner.scan(Path("."))}
    assert out["reader"].summary == "street"


def test_context_rebuilt_from_findings_for_a_partial_run():
    from mitraav.core.context import from_findings, merge
    a = Finding(probe="a", detected=False, meta={"context": {"sound": {"busy": 0.5}}})
    b = Finding(probe="b", detected=False, meta={"context": {"scene": {"kind": "nature"}}})
    assert from_findings([a, b]) == {"sound": {"busy": 0.5}, "scene": {"kind": "nature"}}
    assert merge({"x": 1}, None) == {"x": 1} and merge({"x": 1}, {"x": 2}) == {"x": 2}


def test_probes_work_without_any_context():
    assert run_probe(Reader, Path(".")).summary == "None"
