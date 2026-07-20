"""Measure the frozen joystick's 0.64-s command response for deployment calibration."""
import argparse
import json
from pathlib import Path

import numpy as np

from demo_c.config import TASK
from demo_c.deploy_physics import DEFAULT_JOYSTICK, PhysicsRuntime, quat_yaw, wrap

OUT = Path(__file__).resolve().parent / "out" / "physics" / "bridge_calibration.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joystick", type=Path, default=DEFAULT_JOYSTICK)
    args = parser.parse_args()
    runtime = PhysicsRuntime(args.joystick); rows = []
    velocities = (0.06, 0.14, 0.22, 0.30); turns = (-0.8, -0.4, 0.0, 0.4, 0.8)
    for vx in velocities:
        for vyaw in turns:
            state = runtime.reset(0); rng = runtime.jax.random.key(123)
            # First interval includes stand-start transient; fit the second, settled one.
            state, rng = runtime.run_command(state, np.array([vx, vyaw], np.float32), rng)
            q0 = np.asarray(state.data.qpos); yaw0 = quat_yaw(q0[3:7])
            state, rng = runtime.run_command(state, np.array([vx, vyaw], np.float32), rng)
            q1 = np.asarray(state.data.qpos); delta = q1[:2] - q0[:2]
            c, s = np.cos(-yaw0), np.sin(-yaw0)
            local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
            row = {
                "vx": vx, "vyaw": vyaw, "forward": float(local[0]), "lateral": float(local[1]),
                "turn": wrap(quat_yaw(q1[3:7]) - yaw0), "fell": bool(np.asarray(state.done)),
            }
            rows.append(row); print(row, flush=True)
    good = [row for row in rows if not row["fell"]]
    commanded_forward = np.array([row["vx"] * TASK.step_seconds for row in good])
    actual_forward = np.array([row["forward"] for row in good])
    commanded_turn = np.array([row["vyaw"] * TASK.step_seconds for row in good])
    actual_turn = np.array([row["turn"] for row in good])
    forward_gain = float(commanded_forward @ actual_forward / (commanded_forward @ commanded_forward))
    turn_gain = float(commanded_turn @ actual_turn / (commanded_turn @ commanded_turn))
    result = {"forward_gain": forward_gain, "turn_gain": turn_gain, "rows": rows,
              "joystick": str(args.joystick.resolve()),
              "definition": "actual displacement = gain * commanded velocity * 0.64 s"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"forward_gain": forward_gain, "turn_gain": turn_gain}, indent=2))


if __name__ == "__main__":
    main()
