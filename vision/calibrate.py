"""Interactive calibration CLI implementing ORCHESTRATION.md SS3.5 steps 1-6,
against any transport+camera -- including the sim profile.

    python -m vision.calibrate --profile bench-naked      # interactive
    python -m vision.calibrate --profile sim --auto       # unattended

(Run as a module, from the repo root, so `shared/` resolves on sys.path --
plain `python vision/calibrate.py` will not find the shared/ package.)

Steps, in order (SS3.5):
  1. Axis direction/sign.
  2. Mechanical center offsets (keyboard jog: a/d pan, w/s tilt, enter to
     accept). Interactive only -- this is a physical-hardware step.
  3. Soft limits (jog to safe extremes). Interactive only; provisions the
     ESP32's enforced limits from the same numbers.
  4. Measured deg/px: command +5 deg steps, measure detected-face pixel
     shift, average >= 3 samples.
  5. Deadband/backlash: smallest command with visible motion -> PID
     deadband.
  6. Command->motion latency -> PID gain ceiling.
  7. Audio -- documented stub for speech-engineer (writes audio.json).

Writes profiles/<name>/calibration.json:
    {version: 1, axes: {pan: {sign, center, min, max}, tilt: {...}},
     deg_per_px: {pan, tilt}, deadband_deg, latency_s}

--auto mode runs steps 1/4/5/6 unattended given a SyntheticDetector-
compatible detector or ground_truth_fn (steps 2/3 have no meaningful
unattended analogue on real hardware, so --auto uses center=0/protocol
limits for those -- this is what lets the sim calibrate itself with no
human at the keyboard).
"""
import argparse
import json
import os
import sys
import time

from shared import serial_protocol
from vision.camera import open_camera
from vision.detector import SyntheticDetector
from vision.paths import load_profile_yaml, profile_dir
from vision.transport import open_transport


class Calibrator:
    """Drives one calibration session. All I/O is injected (input_fn,
    print_fn, sleep_fn, clock) so this class is fully unit-testable."""

    def __init__(self, transport, camera, detector, jog_step_deg=2.0,
                 input_fn=input, print_fn=print, sleep_fn=time.sleep,
                 clock=time.monotonic):
        self.transport = transport
        self.camera = camera
        self.detector = detector
        self.jog_step_deg = jog_step_deg
        self._input = input_fn
        self._print = print_fn
        self._sleep = sleep_fn
        self._clock = clock
        self.pan = 0.0
        self.tilt = 0.0

    def _send(self, pan, tilt):
        self.transport.write_line(serial_protocol.encode_target(pan, tilt))

    def _read_center(self, tries=10):
        """(pixel_center, frame_dims) of the highest-score detection, or
        (None, None) if nothing is detected within `tries` attempts."""
        for i in range(tries):
            ok, frame = self.camera.read()
            if ok and frame is not None:
                dets = self.detector.detect(frame)
                if dets:
                    x, y, w, h, _score = max(dets, key=lambda d: d[4])
                    fh, fw = frame.shape[0], frame.shape[1]
                    return (x + w / 2.0, y + h / 2.0), (fw, fh)
            if i < tries - 1:
                self._sleep(0.02)
        return None, None

    # -- Step 1: axis direction/sign -------------------------------------
    def step1_axis_sign(self, auto=False, probe_deg=5.0):
        self._print("== Step 1: axis direction/sign ==")
        pan_sign = self._measure_sign("pan", auto, probe_deg)
        tilt_sign = self._measure_sign("tilt", auto, probe_deg)
        self._print("  pan sign=%+d tilt sign=%+d" % (pan_sign, tilt_sign))
        return pan_sign, tilt_sign

    def _measure_sign(self, axis, auto, probe_deg):
        """Returns the axis's calibrated PID-correction sign: NOT simply
        "which way did the image move" -- its INVERSE, since
        vision/tracking.py uses it as
            err_deg = sign * err_px * deg_per_px
            command = center + PID(err_deg)
        an additive correction toward center, which is only stabilizing
        (negative feedback) if `sign` is the opposite of the observed
        image-shift direction: e.g. if commanding +probe_deg pans the
        image's face LEFT (delta_px < 0, a right-mounted-servo kind of
        convention), then a face sitting to the right of center
        (err_px > 0) needs pan increased (not decreased) to recenter it,
        i.e. sign must be positive despite delta_px having been negative.
        """
        idx = 0 if axis == "pan" else 1
        before, _ = self._read_center()
        if before is None:
            raise RuntimeError(
                "step1: no face detected to measure %s axis sign" % axis)

        if axis == "pan":
            self._send(self.pan + probe_deg, self.tilt)
        else:
            self._send(self.pan, self.tilt + probe_deg)
        self._sleep(0.1)
        after, _ = self._read_center()
        self._send(self.pan, self.tilt)  # revert to baseline

        if not auto:
            self._print("  sent +%.0f deg %s probe" % (probe_deg, axis))

        if after is None:
            raise RuntimeError(
                "step1: lost face while measuring %s axis sign" % axis)

        if auto:
            delta_px = after[idx] - before[idx]
            if delta_px == 0:
                raise RuntimeError(
                    "step1: no measurable pixel shift for %s axis" % axis)
            return -1 if delta_px > 0 else 1

        ans = self._input(
            "Did the %s image shift move in the positive "
            "(right for pan / down for tilt) direction? [y/N] " % axis
        ).strip().lower()
        return -1 if ans.startswith("y") else 1

    # -- Step 2: mechanical center offsets (interactive only) ------------
    def step2_center_offsets(self):
        self._print("== Step 2: mechanical center offsets ==")
        self._print(
            "Jog to the shell's true mechanical center (bracket zero is "
            "not necessarily 1500us neutral)."
        )
        self._jog_loop()
        self._print("  center: pan=%.1f tilt=%.1f" % (self.pan, self.tilt))
        return self.pan, self.tilt

    # -- Step 3: soft limits (interactive only) ---------------------------
    def step3_soft_limits(self):
        self._print("== Step 3: soft limits ==")
        self._print("Jog to the pan MIN safe extreme, blank enter to accept.")
        self._jog_loop()
        pan_min = self.pan
        self._print("Jog to the pan MAX safe extreme, blank enter to accept.")
        self._jog_loop()
        pan_max = self.pan
        self._print("Jog to the tilt MIN safe extreme, blank enter to accept.")
        self._jog_loop()
        tilt_min = self.tilt
        self._print("Jog to the tilt MAX safe extreme, blank enter to accept.")
        self._jog_loop()
        tilt_max = self.tilt

        if pan_min > pan_max:
            pan_min, pan_max = pan_max, pan_min
        if tilt_min > tilt_max:
            tilt_min, tilt_max = tilt_max, tilt_min

        self._print("  pan: [%.1f, %.1f]  tilt: [%.1f, %.1f]"
                     % (pan_min, pan_max, tilt_min, tilt_max))
        return (pan_min, pan_max), (tilt_min, tilt_max)

    def _jog_loop(self):
        while True:
            key = self._input(
                "[a/d pan-/+  w/s tilt-/+  enter=accept] > "
            ).strip().lower()
            if key == "":
                return self.pan, self.tilt
            if key == "a":
                self.pan -= self.jog_step_deg
            elif key == "d":
                self.pan += self.jog_step_deg
            elif key == "w":
                self.tilt -= self.jog_step_deg
            elif key == "s":
                self.tilt += self.jog_step_deg
            else:
                self._print("  (ignored key %r)" % key)
                continue
            self._send(self.pan, self.tilt)
            self._print("  pan=%.1f tilt=%.1f" % (self.pan, self.tilt))

    # -- Step 4: measured deg/px -------------------------------------------
    def step4_deg_per_px(self, auto=False, step_deg=5.0, samples=3,
                          settle_s=0.8):
        """settle_s: wait after commanding the step (and again after
        reverting) long enough for the axis to actually finish moving --
        not just start moving (contrast step6, which measures the first
        sign of motion). A too-short wait here under-measures the true
        pixel shift and over-estimates deg/px, which then makes the real
        PID (err_deg = err_px * deg_per_px) overcorrect against the
        calibrated axis. 0.8s covers this sim's slew/smoothing dynamics
        with margin; real hardware settles a 5deg step well within that."""
        self._print("== Step 4: measured deg/px ==")
        results = {}
        for axis in ("pan", "tilt"):
            idx = 0 if axis == "pan" else 1
            ratios = []
            for i in range(samples):
                if not auto:
                    self._input(
                        "Position a stationary face in frame, then press "
                        "enter to command a +%.0f deg %s step..."
                        % (step_deg, axis))
                before, _ = self._read_center()
                if before is None:
                    raise RuntimeError(
                        "step4: no face detected for %s sample %d"
                        % (axis, i + 1))
                if axis == "pan":
                    self._send(self.pan + step_deg, self.tilt)
                else:
                    self._send(self.pan, self.tilt + step_deg)
                self._sleep(settle_s)
                after, _ = self._read_center()
                self._send(self.pan, self.tilt)  # revert
                # Let the axis actually settle back to baseline before the
                # next sample's "before" read -- otherwise (especially in
                # --auto mode, which has no human-paced pause between
                # samples) "before" can be measured mid-flight from this
                # revert, corrupting the ratio.
                self._sleep(settle_s)
                if after is None:
                    raise RuntimeError(
                        "step4: lost face mid-measurement for %s sample %d"
                        % (axis, i + 1))
                px_shift = after[idx] - before[idx]
                if px_shift == 0:
                    raise RuntimeError(
                        "step4: zero pixel shift measuring %s" % axis)
                ratio = abs(step_deg / px_shift)
                ratios.append(ratio)
                self._print("  %s sample %d: %.4f deg/px"
                             % (axis, i + 1, ratio))
            results[axis] = sum(ratios) / len(ratios)
        self._print("  averaged: pan=%.4f deg/px tilt=%.4f deg/px"
                     % (results["pan"], results["tilt"]))
        return results["pan"], results["tilt"]

    # -- Step 5: deadband/backlash -----------------------------------------
    def step5_deadband(self, auto=False, start_deg=0.05, max_deg=5.0,
                        growth=1.5, px_threshold=1.0):
        self._print("== Step 5: deadband/backlash ==")
        before, _ = self._read_center()
        if before is None:
            raise RuntimeError("step5: no face detected")

        cmd = start_deg
        found = None
        while cmd <= max_deg:
            self._send(self.pan + cmd, self.tilt)
            self._sleep(0.1)
            after, _ = self._read_center()
            self._send(self.pan, self.tilt)
            self._sleep(0.05)
            if after is not None and abs(after[0] - before[0]) >= px_threshold:
                found = cmd
                break
            cmd *= growth

        deadband_deg = found if found is not None else max_deg
        self._print("  deadband_deg=%.3f" % deadband_deg)
        return deadband_deg

    # -- Step 6: command->motion latency ------------------------------------
    def step6_latency(self, auto=False, step_deg=5.0, px_threshold=1.0,
                       poll_interval=0.01, max_wait=2.0):
        self._print("== Step 6: command-to-motion latency ==")
        before, _ = self._read_center()
        if before is None:
            raise RuntimeError("step6: no face detected")

        t0 = self._clock()
        self._send(self.pan + step_deg, self.tilt)
        waited = 0.0
        latency = max_wait
        while waited <= max_wait:
            after, _ = self._read_center(tries=1)
            if after is not None and abs(after[0] - before[0]) >= px_threshold:
                latency = self._clock() - t0
                break
            self._sleep(poll_interval)
            waited += poll_interval
        self._send(self.pan, self.tilt)

        self._print("  latency_s=%.3f" % latency)
        return latency

    # -- Step 7: audio (stub for speech-engineer) ---------------------------
    def step7_audio_stub(self):
        self._print(
            "== Step 7: audio (stub) ==\n"
            "  Speaker level, mic gain + VAD threshold, wake threshold, "
            "self-hearing check.\n"
            "  Owned by speech-engineer (writes profiles/<name>/audio.json); "
            "not performed by vision/calibrate.py."
        )


def write_calibration(name, calibration, root=None):
    path = os.path.join(profile_dir(name, root), "calibration.json")
    with open(path, "w") as f:
        json.dump(calibration, f, indent=2, sort_keys=True)
    return path


FIRMWARE_LIMITS_TEMPLATE = '''\
"""Auto-generated by `python -m vision.calibrate --profile %(name)s` (SS3.5
step 3) -- DO NOT EDIT BY HAND, re-run calibrate.py instead. One set of
soft-limit numbers shared by firmware and Python (ORCHESTRATION.md SS3.5:
"provisions the ESP32's enforced limits -- one set of numbers").

Deploy alongside main.py (see firmware/README.md):
    mpremote cp profiles/%(name)s/firmware_limits.py :firmware_limits.py

firmware/main.py imports this module (flat, at the board root) and falls
back to shared/serial_protocol.py's bench-safe defaults if it is absent.
"""

PAN_MIN = %(pan_min)r
PAN_MAX = %(pan_max)r
TILT_MIN = %(tilt_min)r
TILT_MAX = %(tilt_max)r
'''


def write_firmware_limits(name, calibration, root=None):
    """Writes profiles/<name>/firmware_limits.py: the measured pan/tilt
    soft limits from calibration["axes"], as plain PAN_MIN/PAN_MAX/
    TILT_MIN/TILT_MAX constants -- MicroPython-safe (no dataclasses/f-
    strings/typing), deployable flat to the board root (SS3.5 step 3 /
    firmware/README.md)."""
    axes = calibration["axes"]
    path = os.path.join(profile_dir(name, root), "firmware_limits.py")
    with open(path, "w") as f:
        f.write(FIRMWARE_LIMITS_TEMPLATE % {
            "name": name,
            "pan_min": axes["pan"]["min"], "pan_max": axes["pan"]["max"],
            "tilt_min": axes["tilt"]["min"], "tilt_max": axes["tilt"]["max"],
        })
    return path


def run(profile_name, auto=False, ground_truth_fn=None, transport=None,
        camera=None, detector=None, input_fn=input, print_fn=print,
        sleep_fn=time.sleep, profiles_root=None):
    """Runs steps 1-7 and writes calibration.json. Returns the calibration
    dict. transport/camera/detector are normally opened from the profile,
    but can be injected (tests, --auto against a synthetic world)."""
    profile = load_profile_yaml(profile_name, profiles_root)
    # Camera before transport: sim/world.py's default harness (built lazily
    # on first open_camera(profile) call for camera_source=="sim") binds
    # its SimServoServer's listening socket synchronously, so opening the
    # camera first guarantees that socket is already accepting connections
    # by the time the "socket" serial_transport below tries to connect to
    # it -- letting `--profile sim --auto` work as a single process/command
    # with no separately-launched sim/servo_sim.py server.
    camera = camera or open_camera(profile)
    transport = transport or open_transport(profile)
    if detector is None:
        if ground_truth_fn is not None:
            detector = SyntheticDetector(ground_truth_fn)
        else:
            from vision.detector import YuNetDetector
            detector = YuNetDetector(model_path=profile.get("yunet_model_path"))

    cal = Calibrator(transport, camera, detector, input_fn=input_fn,
                      print_fn=print_fn, sleep_fn=sleep_fn)

    pan_sign, tilt_sign = cal.step1_axis_sign(auto=auto)

    if auto:
        pan_center, tilt_center = 0.0, 0.0
        pan_min, pan_max = serial_protocol.PAN_MIN, serial_protocol.PAN_MAX
        tilt_min, tilt_max = serial_protocol.TILT_MIN, serial_protocol.TILT_MAX
    else:
        pan_center, tilt_center = cal.step2_center_offsets()
        (pan_min, pan_max), (tilt_min, tilt_max) = cal.step3_soft_limits()

    deg_per_px_pan, deg_per_px_tilt = cal.step4_deg_per_px(auto=auto)
    deadband_deg = cal.step5_deadband(auto=auto)
    latency_s = cal.step6_latency(auto=auto)
    cal.step7_audio_stub()

    calibration = {
        "version": 1,
        "axes": {
            "pan": {"sign": pan_sign, "center": pan_center,
                    "min": pan_min, "max": pan_max},
            "tilt": {"sign": tilt_sign, "center": tilt_center,
                     "min": tilt_min, "max": tilt_max},
        },
        "deg_per_px": {"pan": deg_per_px_pan, "tilt": deg_per_px_tilt},
        "deadband_deg": deadband_deg,
        "latency_s": latency_s,
    }
    write_calibration(profile_name, calibration, profiles_root)
    write_firmware_limits(profile_name, calibration, profiles_root)
    return calibration


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CBot calibration (ORCHESTRATION.md SS3.5)")
    parser.add_argument("--profile", default=None,
                         help="defaults to $CBOT_PROFILE, then 'sim'")
    parser.add_argument("--auto", action="store_true",
                         help="unattended mode for steps 1/4/5/6 (sim only)")
    args = parser.parse_args(argv)
    name = args.profile or os.environ.get("CBOT_PROFILE", "sim")

    ground_truth_fn = None
    if args.auto:
        try:
            from sim import world
        except ImportError as exc:
            print(
                "--auto requires a running sim providing ground-truth "
                "bboxes (sim/world.py, owned by sim-engineer): %s" % exc,
                file=sys.stderr,
            )
            return 1
        ground_truth_fn = world.ground_truth_faces

    calibration = run(name, auto=args.auto, ground_truth_fn=ground_truth_fn)
    print("Wrote %s" % os.path.join(profile_dir(name), "calibration.json"))
    print(json.dumps(calibration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
