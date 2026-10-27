"""Physics sanity tests: boxes whose behavior has a textbook answer, run through the same probes in every engine.

Push at height hp on a box with half-width b (along the push), fingertip friction mu_p = mu / 2: the fingertip's
friction pins the near side down, so the box tips only if mu > b / (hp - mu_p * b); otherwise it slides.
Slowly tilting ramp: it slips at atan(mu) or tips at atan(b / h_com), whichever angle comes first.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sim  # noqa: E402

BOXES = [  # (name, x, y, z size in m, mu)
    ("squat box", 0.10, 0.064, 0.060, 0.4),
    ("tall box", 0.04, 0.04, 0.15, 0.5),
    ("tall grippy box", 0.04, 0.04, 0.15, 0.9),
    ("flat tile", 0.20, 0.20, 0.02, 0.6),
]
PROBES = [("push", {"h": 0.2, "dir": 90}), ("push", {"h": 0.85, "dir": 90}), ("ramp", {"dir": 90})]


def box_record(name, x, y, z, mu, m=0.5):
    c = np.array([[sx * x / 2, sy * y / 2, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    I = m / 12 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    return {"name": name, "dims": [x, y, z], "hull": c.tolist(), "mass": m, "com": [0, 0, z / 2], "inertia": I.tolist(),
            "friction": mu, "restitution": 0.0}


def expected(kind, p, x, y, z, mu):
    b = y / 2  # pushes and ramps here act along y
    if kind == "push":
        hp = p["h"] * z
        lever = hp - mu / 2 * b
        return "tips" if lever > 0 and mu > b / lever else "slides"
    slip, tip = math.degrees(math.atan(mu)), math.degrees(math.atan(b / (z / 2)))
    if min(slip, tip) > 45:
        return "holds"
    return "tips" if tip < slip else "slides"


def observed(res):
    f = res[0]
    tilt = np.degrees(np.linalg.norm(f[:, 3:6], axis=1)).max()
    disp = np.linalg.norm(f[-1, 0:2])
    return "tips" if tilt > 30 else ("slides" if disp > 0.05 else "holds"), tilt, disp


if __name__ == "__main__":
    rows, ok = [], {"mujoco": 0, "physx": 0}
    n = 0
    for name, x, y, z, mu in BOXES:
        a = box_record(name, x, y, z, mu)
        for kind, p in PROBES:
            want = expected(kind, p, x, y, z, mu)
            got = {}
            for eng in ("mujoco", "physx"):
                rig = sim.make_rig(a, ".", eng, "union")
                got[eng] = observed(sim.Prober(rig).run(kind, p, 0.0))
                ok[eng] += got[eng][0] == want
            n += 1
            label = f"{kind} h={p['h']}" if kind == "push" else "ramp"
            rows.append((name, label, want, got))
            print(f"{name:16s} {label:11s} expect {want:6s} | MuJoCo {got['mujoco'][0]:6s} (tilt {got['mujoco'][1]:5.1f}°) "
                  f"| PhysX {got['physx'][0]:6s} (tilt {got['physx'][1]:5.1f}°)")
    print({k: f"{v}/{n} match the textbook answer" for k, v in ok.items()})
    json.dump({"cases": [(r[0], r[1], r[2], {k: v[0] for k, v in r[3].items()}) for r in rows], "score": ok, "n": n},
              open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
