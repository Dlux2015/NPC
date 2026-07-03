import json

import numpy as np
import pytest

from shared import serial_protocol
from vision import calibrate
from vision.detector import SyntheticDetector
from vision.tracking import CalibrationError, load_profile


def _write_profile(tmp_path, name="testcal", serial_transport="socket",
                    camera_source="sim"):
    pdir = tmp_path / "profiles" / name
    pdir.mkdir(parents=True)
    (pdir / "profile.yaml").write_text(
        "name: %s\nserial_transport: %s\ncamera_source: %s\n"
        % (name, serial_transport, camera_source)
    )
    return tmp_path / "profiles", pdir


class _LinearWorld:
    """Minimal deterministic pan/tilt -> pixel-space model so calibrate.py's
    --auto measurement algorithms can be exercised without sim/world.py
    (owned by sim-engineer). Reacts instantly to commands -- good enough to
    prove the calibration math, not to model real servo latency."""

    def __init__(self, deg_per_px=0.05, size=(640, 480)):
        self.pan = 0.0
        self.tilt = 0.0
        self.deg_per_px = deg_per_px
        self.w, self.h = size

    def send(self, line):
        parsed = serial_protocol.parse_line(line)
        if parsed and parsed[0] == "target":
            self.pan, self.tilt = parsed[1], parsed[2]

    def ground_truth(self, frame):
        cx = self.w / 2.0 + self.pan / self.deg_per_px
        cy = self.h / 2.0 + self.tilt / self.deg_per_px
        return [(cx - 50, cy - 50, 100.0, 100.0, 0.95)]

    def frame(self):
        return np.zeros((self.h, self.w, 3), dtype=np.uint8)


class _WorldTransport:
    def __init__(self, world):
        self.world = world

    def write_line(self, line):
        self.world.send(line)


class _WorldCamera:
    def __init__(self, world):
        self.world = world

    def read(self):
        return True, self.world.frame()


def test_auto_calibration_round_trip(tmp_path):
    profiles_root, pdir = _write_profile(tmp_path)
    world = _LinearWorld(deg_per_px=0.05)
    detector = SyntheticDetector(world.ground_truth)

    calibration = calibrate.run(
        "testcal", auto=True, detector=detector,
        transport=_WorldTransport(world), camera=_WorldCamera(world),
        sleep_fn=lambda s: None, profiles_root=str(profiles_root),
    )

    assert calibration["version"] == 1
    assert set(calibration["axes"]) == {"pan", "tilt"}
    for axis in ("pan", "tilt"):
        a = calibration["axes"][axis]
        assert set(a) == {"sign", "center", "min", "max"}
        assert a["sign"] in (1, -1)
        assert a["min"] < a["max"]
    assert calibration["deg_per_px"]["pan"] == pytest.approx(0.05, rel=0.2)
    assert calibration["deg_per_px"]["tilt"] == pytest.approx(0.05, rel=0.2)
    assert calibration["deadband_deg"] >= 0
    assert calibration["latency_s"] >= 0

    written_path = pdir / "calibration.json"
    assert written_path.exists()
    on_disk = json.loads(written_path.read_text())
    assert on_disk == calibration

    # F7: one set of soft-limit numbers for firmware too.
    limits_path = pdir / "firmware_limits.py"
    assert limits_path.exists()
    limits_ns = {}
    exec(compile(limits_path.read_text(), str(limits_path), "exec"), limits_ns)
    assert limits_ns["PAN_MIN"] == calibration["axes"]["pan"]["min"]
    assert limits_ns["PAN_MAX"] == calibration["axes"]["pan"]["max"]
    assert limits_ns["TILT_MIN"] == calibration["axes"]["tilt"]["min"]
    assert limits_ns["TILT_MAX"] == calibration["axes"]["tilt"]["max"]


def test_auto_calibration_writes_versioned_file_per_profile(tmp_path):
    profiles_root, pdir = _write_profile(tmp_path, name="another")
    world = _LinearWorld(deg_per_px=0.1)
    detector = SyntheticDetector(world.ground_truth)

    calibrate.run(
        "another", auto=True, detector=detector,
        transport=_WorldTransport(world), camera=_WorldCamera(world),
        sleep_fn=lambda s: None, profiles_root=str(profiles_root),
    )

    profile, calibration = load_profile("another", root=str(profiles_root))
    assert profile["name"] == "another"
    assert calibration["version"] == 1


def test_tracking_refuses_to_start_without_calibration(tmp_path):
    profiles_root, _pdir = _write_profile(tmp_path, name="nocalib")
    with pytest.raises(CalibrationError, match="calibrate.py"):
        load_profile("nocalib", root=str(profiles_root))


def test_tracking_refuses_missing_profile_dir_too(tmp_path):
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    with pytest.raises(FileNotFoundError):
        load_profile("ghost", root=str(profiles_root))
