import argparse
import array
import base64
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_script("benchmark_worker")
compare = load_script("compare_benchmarks")


def wire(values, shape=None):
    floats = array.array("d", (part for value in values for part in (complex(value).real, complex(value).imag)))
    return [base64.b64encode(floats.tobytes()).decode(), shape if shape is not None else [len(values)]]


def run(values, shape=None):
    return {"outputs": {"pressure@100.000": wire(values, shape)}}


def test_windows_report_preserves_unavailable_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.delattr(os, "uname", raising=False)
    monkeypatch.setattr(benchmark.PeakMemory, "_reader", lambda self: lambda: None)
    with benchmark.PeakMemory(os.getpid()) as memory:
        assert memory.current_mb() is None
    assert memory.peak_mb is None
    args = argparse.Namespace(
        label="test", backend="cpu", precision="float32", request=None, mesh="test.msh", threads="2"
    )
    out = tmp_path / "run.json"
    benchmark.write_run(out, args, "test", 0, 0, 1, 1, [], None, memory.peak_mb)
    record = json.loads(out.read_text())
    assert record["memory_mb"] == {"before_sweep": None, "peak_sweep": None}
    assert record["host"] == benchmark.platform.machine()
    assert "peak unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("actual", [run([1]), run([1], [2]), run([1, 2], [1, 2]), {"outputs": {}}])
def test_accuracy_rejects_missing_or_truncated_outputs(actual):
    with pytest.raises(ValueError, match="shape|length|quantities"):
        compare.accuracy(actual, run([1, 2]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_accuracy_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        compare.accuracy(run([value]), run([1]))
    with pytest.raises(ValueError, match="non-finite"):
        compare.accuracy(run([1]), run([value]))


def test_accuracy_handles_zero_signals():
    assert compare.accuracy(run([0]), run([1]))["pressure"] == (1.0, math.inf)
    assert compare.accuracy(run([1]), run([0]))["pressure"][0] == math.inf
    assert compare.accuracy(run([0]), run([0]))["pressure"] == (0.0, 0.0)


def test_accuracy_measures_complex_phase():
    relative, db = compare.accuracy(run([1j]), run([1]))["pressure"]
    assert relative == pytest.approx(math.sqrt(2))
    assert db == 0


def test_stage_checks_later_repeats():
    ref = run([1])
    relative, db = compare.stage_accuracy([ref, run([2])], ref)["pressure"]
    assert relative == 1
    assert db == pytest.approx(20 * math.log10(2))
    with pytest.raises(ValueError, match="run 2"):
        compare.stage_accuracy([ref, run([])], ref)


def test_cli_compares_all_repeats_with_unavailable_memory(tmp_path, monkeypatch, capsys):
    for index, value in enumerate([1, 2]):
        record = {
            **run([value]),
            "wall_s": 1.0,
            "commit": "test",
            "backend": "cpu",
            "memory_mb": {"before_sweep": None, "peak_sweep": None},
            "frequencies": [],
        }
        (tmp_path / f"run{index}.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        sys, "argv", ["compare", "--reference", str(tmp_path / "run0.json"), "--stage", f"cpu={tmp_path / 'run*.json'}"]
    )
    compare.main()
    output = capsys.readouterr().out
    assert "memory  unavailable" in output
    assert "pressure 1.0e+00 rel, 6.021 dB" in output


def test_cli_rejects_invalid_later_repeat(tmp_path, monkeypatch):
    reference = {
        **run([1]),
        "wall_s": 1.0,
        "commit": "test",
        "backend": "cpu",
        "memory_mb": {"before_sweep": None, "peak_sweep": None},
        "frequencies": [],
    }
    (tmp_path / "run0.json").write_text(json.dumps(reference))
    (tmp_path / "run1.json").write_text(json.dumps({**reference, "outputs": {}}))
    monkeypatch.setattr(
        sys, "argv", ["compare", "--reference", str(tmp_path / "run0.json"), "--stage", f"cpu={tmp_path / 'run*.json'}"]
    )
    with pytest.raises(SystemExit) as exc:
        compare.main()
    assert exc.value.code == 2


matrix = load_script("benchmark_matrix")
porous_requests = load_script("porous_benchmark_requests")


def test_matrix_interleaves_rounds_and_compares_in_config_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    other = tmp_path / "main" / "src"
    argv = ["matrix", "--backend", "cuda", "--runs", "2", "--out-dir", str(tmp_path / "runs")]
    argv += ["--config", f"main=req.json@{other}", "--config", "JCA limp=req.json"]
    monkeypatch.setattr(sys, "argv", argv)
    matrix.main()
    runs, (compare_command, _) = calls[:-1], calls[-1]
    outs = [Path(command[command.index("--out") + 1]).name for command, _ in runs]
    assert outs == ["main_r1.json", "JCA_limp_r1.json", "main_r2.json", "JCA_limp_r2.json"]
    assert all(command[command.index("--backend") + 1] == "cuda" for command, _ in runs)
    assert all(Path(command[command.index("--out") + 1]).is_absolute() for command, _ in runs)
    assert runs[0][1]["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(other.resolve())
    assert runs[1][1]["env"].get("PYTHONPATH") == os.environ.get("PYTHONPATH")
    stages = [compare_command[i + 1] for i, arg in enumerate(compare_command) if arg == "--stage"]
    assert [stage.split("=")[0] for stage in stages] == ["main", "JCA limp"]
    assert stages[1].endswith("JCA_limp_r*.json")


def test_matrix_config_keeps_windows_drive_letters():
    label, request, source = matrix.parse_config(r"main=C:\runs\sawmod.json@C:\src\BEAT_main\src")
    assert label == "main"
    assert str(request).endswith(r"C:\runs\sawmod.json") or request.name == "sawmod.json"
    assert str(source).endswith("src")


def test_porous_variants_change_only_the_body_material(tmp_path, monkeypatch):
    regions = [
        {"id": "region:chamber", "kind": "bounded_air", "loss_model": {"bulk_loss_factor": 0.02}},
        {"id": "region:port", "kind": "bounded_air", "loss_model": {"bulk_loss_factor": 0.01}},
    ]
    request = {"compiled_system": {"regions": regions}, "solver_options": {"precision": "float32"}}
    (tmp_path / "box.json").write_text(json.dumps(request))
    argv = ["porous", str(tmp_path / "box.json"), "--body", "region:chamber", "--out-dir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    porous_requests.main()
    loss = {}
    for name in ("air", "jca", "jcal", "limp"):
        variant = json.loads((tmp_path / "out" / f"box_{name}.json").read_text())
        assert variant["solver_options"] == {"precision": "float32", "phasor_convention": "exp(+i omega t)"}
        chamber, port = variant["compiled_system"]["regions"]
        assert port == regions[1]
        loss[name] = chamber["loss_model"]
    assert loss["air"] == {"bulk_loss_factor": 0.02}
    assert (loss["jca"]["model"], loss["jca"]["frame"], loss["jca"]["porosity"]) == ("jca", "rigid", 0.98)
    assert loss["jcal"]["static_thermal_permeability_m2"] == 2e-9 and loss["jcal"]["model"] == "jcal"
    assert loss["limp"]["frame"] == "limp" and loss["limp"]["frame_density_kg_per_m3"] == 30.0
