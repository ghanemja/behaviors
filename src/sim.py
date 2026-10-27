"""Run a standard, size-normalized probe battery on each converted asset, in MuJoCo or PhysX.

Engines and collision geometry:
  --engine mujoco --geom parts   one convex hull per authored collision part (NVIDIA's MuJoCo variant: convexHull)
  --engine mujoco --geom union   one convex hull around all collision parts (the first version of this project)
  --engine physx  --geom parts   PhysX 5 with the same per-part convex hulls (isolates the engine)
  --engine physx  --geom sdf     PhysX 5 with SDF triangle-mesh collision per part (NVIDIA's PhysX variant: sdf)
The probe logic, timing, size normalization and recorded features are shared by all of them.
"""
import argparse
import glob
import json
import math
import os
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.spatial import ConvexHull

G = 9.81
NF = 100  # frames recorded per rollout
DT = 0.002

TRAIN_PROBES = [
    ("push", {"h": 0.2, "dir": 0}), ("push", {"h": 0.2, "dir": 90}),
    ("push", {"h": 0.5, "dir": 0}), ("push", {"h": 0.5, "dir": 90}),
    ("push", {"h": 0.85, "dir": 0}), ("push", {"h": 0.85, "dir": 90}),
    ("ramp", {"dir": 0}), ("ramp", {"dir": 90}),
    ("drop", {"axis": 0, "tilt": 25}), ("drop", {"axis": 90, "tilt": 25}),
]
HELDOUT_PROBES = [
    ("push", {"h": 0.65, "dir": 45, "speed": 1.4}),
    ("ramp", {"dir": 45}),
    ("drop", {"axis": 45, "tilt": 45, "height": 1.0}),
]
REPEATS = 3


def quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) or 1)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q2R(q):
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) or 1)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rotvec(R):
    q = R2q(R)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return np.zeros(3)
    return 2 * math.atan2(s, q[0]) * q[1:] / s


def hull_vertices(v, S, max_verts=128):
    """Convex hull vertices of a collision part, reduced to at most max_verts here so that MuJoCo and PhysX get the
    identical polytope (each engine would otherwise simplify large hulls with its own algorithm). Flat parts get a
    hair of thickness so the hull is valid."""
    import fast_simplification
    v = np.asarray(v, float)
    try:
        h = ConvexHull(v)
        ok = h.volume > 1e-12 * S ** 3
    except Exception:
        ok = False
    if not ok:
        c = v.mean(0)
        _, _, vt = np.linalg.svd(v - c, full_matrices=False)
        n = vt[-1] * max(1e-3 * S, 1e-4)
        v = np.vstack([v + n, v - n])
        h = ConvexHull(v, qhull_options="QJ")
    for _ in range(6):
        if len(h.vertices) <= max_verts:
            break
        idx = {j: i for i, j in enumerate(h.vertices)}
        hv = v[h.vertices].astype(np.float32)
        hf = np.array([[idx[j] for j in tri] for tri in h.simplices], np.int32)
        target = 1 - (2 * max_verts - 8) / len(hf)
        sv, _ = fast_simplification.simplify(hv, hf, target_reduction=float(np.clip(target, 0.0, 0.99)))
        v = np.asarray(sv, float)
        h = ConvexHull(v)
    return v[h.vertices]


def load_parts(a, assets_dir, geom):
    """Collision parts as (vertices, faces) in the asset frame; 'union' collapses them to one hull."""
    S = max(a["dims"])
    if geom == "union":
        return [(hull_vertices(a["hull"], S), None)]
    z = np.load(f"{assets_dir}/{a['name']}.parts.npz")
    parts = []
    for v0, nv, f0, nf, _ in z["meta"]:
        v = z["v"][v0:v0 + nv].astype(float)
        f = z["f"][f0:f0 + nf] - 0
        parts.append((v, f))
    if geom in ("parts",):
        parts = [(hull_vertices(v, S), None) for v, _ in parts]
    elif geom == "sdf":
        parts = [cap_faces(v, f) for v, f in parts]
    return parts


SDF_MAX_FACES = 20000  # finer than the SDF grid (S/128); keeps SDF cooking time bounded


def cap_faces(v, f):
    if len(f) <= SDF_MAX_FACES:
        return v, f
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(v, np.float64), np.asarray(f, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=SDF_MAX_FACES, preservetopology=False, qualitythr=0.3)
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix())


def principal(I):
    w, V = np.linalg.eigh(np.asarray(I, float))
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    return np.maximum(w, 1e-12), V


# ---------------------------------------------------------------- MuJoCo backend
class MujocoRig:
    def __init__(self, a, parts):
        import mujoco
        self.mj = mujoco
        self.a, self.S = a, max(a["dims"])
        self.hull = np.asarray(a["hull"], float)
        floor_half = max(3.0, 8 * self.S)
        I = np.asarray(a["inertia"])
        com = a["com"]
        rest = min(max(a["restitution"], 0.0), 0.95)
        zeta = 1.0 if rest < 1e-3 else -math.log(rest) / math.sqrt(math.pi ** 2 + math.log(rest) ** 2)
        mu = max(a["friction"], 0.02)
        r = max(0.03 * self.S, 0.003)
        meshes = "".join(f'<mesh name="p{k}" vertex="{" ".join(f"{x:.6f}" for x in np.asarray(v).reshape(-1))}" maxhullvert="-1" inertia="convex"/>'
                         for k, (v, _) in enumerate(parts))
        geoms = "".join(f'<geom name="obj{k}" type="mesh" mesh="p{k}" friction="{mu} 0.005 0.0001" solref="0.005 {zeta:.3f}" condim="4"/>'
                        for k in range(len(parts)))
        xml = f"""
<mujoco model="probe">
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <asset>{meshes}</asset>
  <worldbody>
    <body name="floor" mocap="true">
      <geom name="floor" type="box" size="{floor_half} {floor_half} 0.05" pos="0 0 -0.05" friction="0 0.005 0.0001"/>
    </body>
    <body name="obj">
      <freejoint/>
      <inertial pos="{com[0]} {com[1]} {com[2]}" mass="{a['mass']}" fullinertia="{I[0,0]} {I[1,1]} {I[2,2]} {I[0,1]} {I[0,2]} {I[1,2]}"/>
      {geoms}
    </body>
    <body name="pusher" mocap="true" pos="0 0 -10">
      <geom name="pusher" type="sphere" size="{r}" friction="{mu / 2} 0.0001 0.0001" priority="1"/>
    </body>
  </worldbody>
</mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.bid = self.m.body("obj").id
        self.floor_mid = self.m.body("floor").mocapid[0]
        self.push_mid = self.m.body("pusher").mocapid[0]
        self.r = float(self.m.geom("pusher").size[0])
        self.hidden = np.array([0, 0, -10 - 10 * self.S])

    def reset(self, quat, lift=0.0):
        mj, m, d = self.mj, self.m, self.d
        mj.mj_resetData(m, d)
        d.mocap_pos[self.floor_mid] = 0
        d.mocap_quat[self.floor_mid] = [1, 0, 0, 0]
        d.mocap_pos[self.push_mid] = self.hidden
        R = q2R(quat)
        hv = self.hull @ R.T
        com_w = R @ np.asarray(self.a["com"])
        d.qpos[:3] = [-com_w[0], -com_w[1], -hv[:, 2].min() + 1e-4 + lift]
        d.qpos[3:7] = quat
        mj.mj_forward(m, d)

    def settle(self, seconds):
        for _ in range(int(seconds / DT)):
            self.mj.mj_step(self.m, self.d)
        self.d.qvel[:] = 0
        com = self.d.xipos[self.bid].copy()
        self.d.qpos[:2] -= com[:2]
        self.mj.mj_forward(self.m, self.d)

    def set_pusher(self, pos):
        self.d.mocap_pos[self.push_mid] = self.hidden if pos is None else pos

    def set_floor(self, q):
        self.d.mocap_quat[self.floor_mid] = q

    def step(self):
        self.mj.mj_step(self.m, self.d)

    def quat(self):
        return self.d.qpos[3:7].copy()

    def state(self):
        vel = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY, self.bid, vel, 0)
        return (self.d.xmat[self.bid].reshape(3, 3).copy(), self.d.xipos[self.bid].copy(), self.d.xpos[self.bid].copy(),
                vel[3:].copy(), vel[:3].copy())

    def pusher_pos(self):
        return self.d.mocap_pos[self.push_mid].copy()

    def floor_quat(self):
        return self.d.mocap_quat[self.floor_mid].copy()


# ---------------------------------------------------------------- PhysX backend (SAPIEN 3, PhysX 5 on CPU)
class PhysxRig:
    def __init__(self, a, parts, sdf):
        import sapien
        from sapien import physx
        self.sp, self.px = sapien, physx
        self.a, self.S = a, max(a["dims"])
        S = self.S
        self.hull = np.asarray(a["hull"], float)
        cfg = physx.PhysxSceneConfig()
        cfg.enable_tgs = True
        cfg.enable_pcm = True
        cfg.bounce_threshold = 0.2
        cfg.gravity = np.array([0, 0, -G], np.float32)
        physx.set_scene_config(cfg)
        sh = physx.PhysxShapeConfig()
        sh.contact_offset = max(0.02 * S, 0.001)
        sh.rest_offset = 0.0
        physx.set_shape_config(sh)
        if sdf:
            sc = physx.PhysxSDFConfig()
            sc.spacing = S / 128
            sc.subgridSize = 6
            sc.num_threads_for_construction = 1
            physx.set_sdf_config(sc)
        self.scene = sapien.Scene([physx.PhysxCpuSystem()])
        self.scene.set_timestep(DT)
        mu = max(a["friction"], 0.02)
        e = min(max(a["restitution"], 0.0), 0.95)
        mat = physx.PhysxMaterial(mu, mu, e)
        # floor and pusher are kinematic; the floor shares the object's material so PhysX's averaging leaves mu and e unchanged
        self.floor = self._kinematic(physx.PhysxCollisionShapeBox(np.array([max(3.0, 8 * S)] * 2 + [0.05], np.float32), mat),
                                     sapien.Pose([0, 0, -0.05]))
        self.r = max(0.03 * S, 0.003)
        # PhysX averages friction, so a frictionless pusher gives mu/2 at the fingertip; MuJoCo's pusher is set to mu/2 to match
        self.pusher = self._kinematic(physx.PhysxCollisionShapeSphere(self.r, physx.PhysxMaterial(0.0, 0.0, 0.0)), sapien.Pose())
        self.hidden = np.array([0, 0, -10 - 10 * S])
        ent = sapien.Entity()
        body = physx.PhysxRigidDynamicComponent()
        one = np.ones(3, np.float32)
        for v, f in parts:
            if sdf:
                shape = physx.PhysxCollisionShapeTriangleMesh(np.asarray(v, np.float32), np.asarray(f, np.uint32), one, mat, sdf=True)
            else:
                shape = physx.PhysxCollisionShapeConvexMesh(np.asarray(v, np.float32), one, mat)
            body.attach(shape)
        ent.add_component(body)
        self.scene.add_entity(ent)
        body.set_mass(float(a["mass"]))
        w, V = principal(a["inertia"])
        body.set_cmass_local_pose(sapien.Pose(np.asarray(a["com"], np.float32), R2q(V).astype(np.float32)))
        body.set_inertia(w.astype(np.float32))
        body.set_sleep_threshold(0.0)
        body.set_solver_position_iterations(64)  # 16 leaves pusher/floor friction unconverged and invents tips (see sanity.py)
        body.set_solver_velocity_iterations(1)
        assert abs(body.get_mass() - a["mass"]) <= 1e-4 * a["mass"], "mass not applied"
        self.ent, self.body = ent, body
        self.com_local = np.asarray(a["com"], float)

    def _kinematic(self, shape, local):
        ent = self.sp.Entity()
        c = self.px.PhysxRigidDynamicComponent()
        shape.local_pose = local
        c.attach(shape)
        c.kinematic = True
        ent.add_component(c)
        self.scene.add_entity(ent)
        return c

    def _place_kinematic(self, comp, p, q):
        pose = self.sp.Pose(np.asarray(p, np.float32), np.asarray(q, np.float32))
        comp.entity.set_pose(pose)
        comp.kinematic_target = pose

    def reset(self, quat, lift=0.0):
        self._place_kinematic(self.floor, [0, 0, 0], [1, 0, 0, 0])
        self._place_kinematic(self.pusher, self.hidden, [1, 0, 0, 0])
        R = q2R(quat)
        hv = self.hull @ R.T
        com_w = R @ self.com_local
        p = np.array([-com_w[0], -com_w[1], -hv[:, 2].min() + 1e-4 + lift])
        self.ent.set_pose(self.sp.Pose(p.astype(np.float32), np.asarray(quat, np.float32)))
        self.body.linear_velocity = np.zeros(3, np.float32)
        self.body.angular_velocity = np.zeros(3, np.float32)

    def settle(self, seconds):
        for _ in range(int(seconds / DT)):
            self.scene.step()
        R, com, pos, _, _ = self.state()
        p = pos.copy()
        p[:2] -= com[:2]
        self.ent.set_pose(self.sp.Pose(p.astype(np.float32), self.ent.pose.q))
        self.body.linear_velocity = np.zeros(3, np.float32)
        self.body.angular_velocity = np.zeros(3, np.float32)

    def set_pusher(self, pos):
        target = self.hidden if pos is None else np.asarray(pos, float)
        if np.linalg.norm(target - self.pusher_pos()) > 0.25 * self.S:
            # a large jump (parking spot <-> start) is a teleport; a kinematic target would turn it into a huge velocity
            self._place_kinematic(self.pusher, target, [1, 0, 0, 0])
        else:
            self.pusher.kinematic_target = self.sp.Pose(target.astype(np.float32))

    def set_floor(self, q):
        self.floor.kinematic_target = self.sp.Pose([0, 0, 0], np.asarray(q, np.float32))

    def step(self):
        self.scene.step()

    def quat(self):
        return np.asarray(self.ent.pose.q, float)

    def state(self):
        pose = self.ent.pose
        R = q2R(pose.q)
        pos = np.asarray(pose.p, float)
        return R, pos + R @ self.com_local, pos, np.asarray(self.body.linear_velocity, float), np.asarray(self.body.angular_velocity, float)

    def pusher_pos(self):
        return np.asarray(self.pusher.entity.pose.p, float)

    def floor_quat(self):
        return np.asarray(self.floor.entity.pose.q, float)


# ---------------------------------------------------------------- shared probe logic
def make_rig(a, assets_dir, engine, geom):
    parts = load_parts(a, assets_dir, "union" if geom == "union" else geom)
    if engine == "mujoco":
        return MujocoRig(a, parts)
    return PhysxRig(a, parts, sdf=(geom == "sdf"))


class Prober:
    def __init__(self, rig):
        self.rig = rig
        self.S = rig.S
        self.T = float(np.clip(30 * math.sqrt(self.S / G), 1.5, 8.0))
        self.vref = math.sqrt(G * self.S)

    def lowest(self, Rw, pw, floor_R):
        hv = (self.rig.hull @ Rw.T) + pw
        return (hv @ floor_R)[:, 2].min()

    def run(self, kind, p, yaw, speed_mult=1.0):
        rig, S = self.rig, self.S
        dir_ang = math.radians(p.get("dir", p.get("axis", 0)))
        e1 = np.array([math.cos(dir_ang), math.sin(dir_ang), 0.0])
        e3 = np.array([0, 0, 1.0])
        e2 = np.cross(e3, e1)
        B = np.stack([e1, e2, e3])  # world -> probe frame
        q_yaw = quat_axis_angle([0, 0, 1], yaw)
        rig.reset(q_yaw)
        rig.settle(0.6)
        if kind == "drop":
            q_now = rig.quat()
            R_up = q2R(q_now)
            q0 = qmul(quat_axis_angle(e2, math.radians(p["tilt"])), q_now)
            rig.reset(q0, lift=p.get("height", 0.5) * S)
        R_now, start_pos, body_pos, _, _ = rig.state()
        R0 = R_up if kind == "drop" else R_now
        steps = int(self.T / DT)
        every = max(steps // NF, 1)
        feats, play, floor_ang = [], [], []
        th = 0.0
        if kind == "push":
            hv = (rig.hull @ R0.T) + body_pos
            proj = hv @ e1
            H = hv[:, 2].max() - hv[:, 2].min()
            start = start_pos.copy()
            start[2] = hv[:, 2].min() + p["h"] * H
            start += e1 * (proj.min() - start_pos @ e1 - rig.r - 0.02 * S)
            v = 0.1 * self.vref * speed_mult * p.get("speed", 1.0)
            travel = 0.6 * S + rig.r + 0.02 * S
            rig.set_pusher(start)
        ramp_rate = math.radians(45) / (0.8 * self.T)
        done = False
        for k in range(steps):
            t = k * DT
            if done:
                if k % every == 0 and len(feats) < NF:
                    last = feats[-1].copy()
                    last[6:12] = 0
                    feats.append(last)
                    play.append(play[-1])
                    floor_ang.append(floor_ang[-1])
                continue
            if kind == "push":
                rig.set_pusher(start + e1 * min(v * t, travel) if t < travel / v * 1.3 else None)
            elif kind == "ramp":
                th = min(ramp_rate * t, math.radians(45))
                rig.set_floor(quat_axis_angle(np.cross(e3, e1), th))
            rig.step()
            if k % every == 0 and len(feats) < NF:
                Rw, pw, xpos, lv, w = rig.state()
                if kind == "ramp":
                    Rf = q2R(rig.floor_quat())
                    floor_ang.append(math.degrees(th))
                else:
                    Rf = np.eye(3)
                Rrel = Rf.T @ Rw @ R0.T
                prel = Rf.T @ pw - start_pos
                zmin = self.lowest(Rw, xpos, Rf)
                f = np.r_[B @ prel / S, B @ rotvec(Rrel), B @ lv / self.vref, B @ w * math.sqrt(S / G),
                          zmin / S, (Rf.T @ pw)[2] / S]
                feats.append(f)
                pp = rig.pusher_pos() if kind == "push" else np.zeros(3)
                play.append(np.r_[xpos, R2q(Rw), pp, rig.floor_quat()])
                if kind == "ramp" and np.linalg.norm(prel[:2]) > 1.5 * S:
                    done = True
        feats = np.array(feats, dtype=np.float32)
        if not np.isfinite(feats).all() or np.abs(feats[:, 6:9]).max() > 40:
            return None
        return feats, np.array(play, dtype=np.float32), np.array(floor_ang, dtype=np.float32)


def outcomes(kind, f, floor_ang):
    tilt = np.degrees(np.linalg.norm(f[:, 3:6], axis=1))
    disp = np.linalg.norm(f[:, 0:2], axis=1)
    speed = np.linalg.norm(f[:, 6:9], axis=1)
    moving = np.where(speed > 0.05)[0]
    settle = (moving[-1] + 1) / len(f) if len(moving) else 0.0
    out = {"max_tilt": float(tilt.max()), "final_tilt": float(tilt[-1]), "disp": float(disp[-1]),
           "max_speed": float(speed.max()), "settle": float(settle), "toppled": float(tilt[-1] > 45)}
    if kind == "ramp":
        slip = np.where(disp > 0.05)[0]
        tip = np.where(tilt > 15)[0]
        out["slip_angle"] = float(floor_ang[slip[0]]) if len(slip) else 50.0
        out["tip_angle"] = float(floor_ang[tip[0]]) if len(tip) else 50.0
    if kind == "drop":
        contact = f[:, 12] < 0.01
        out["bounces"] = float(np.sum(contact[:-1] & ~contact[1:]))
    return out


def process(job):
    path, out_dir, engine, geom, repeats = job
    a = json.load(open(path))
    name = a["name"]
    if os.path.exists(f"{out_dir}/{name}.json"):
        return name, None
    try:
        rig = make_rig(a, os.path.dirname(path), engine, geom)
    except Exception as e:
        return name, f"model error: {type(e).__name__} {str(e)[:120]}"
    pr = Prober(rig)
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # stable across runs and machines
    rig.reset(np.array([1.0, 0, 0, 0]))
    rig.settle(1.0)
    rest_tilt = math.degrees(np.linalg.norm(rotvec(rig.state()[0])))
    train, train_out, play = [], [], {}
    for pi, (kind, p) in enumerate(TRAIN_PROBES):
        reps = []
        for r in range(repeats):
            yaw = 0.0 if r == 0 else rng.uniform(-0.25, 0.25)
            res = pr.run(kind, p, yaw, speed_mult=1.0 if r == 0 else rng.uniform(0.85, 1.15))
            if res is None:
                return name, f"unstable in probe {kind}{p}"
            feats, pl, fa = res
            reps.append(feats)
            train_out.append((pi, outcomes(kind, feats, fa)))
            if r == 0:
                play[pi] = pl
        train.append(reps)
    held, held_out = [], []
    for kind, p in HELDOUT_PROBES:
        res = pr.run(kind, p, 0.0)
        if res is None:
            return name, f"unstable in heldout {kind}"
        feats, pl, fa = res
        held.append(feats)
        held_out.append(outcomes(kind, feats, fa))
    np.savez_compressed(f"{out_dir}/{name}.npz", train=np.array(train), held=np.array(held),
                        play=np.array([play[i] for i in range(len(TRAIN_PROBES))]))
    json.dump({"name": name, "T": pr.T, "S": pr.S, "rest_tilt": rest_tilt, "engine": engine, "geom": geom,
               "train_out": train_out, "held_out": held_out}, open(f"{out_dir}/{name}.json", "w"))
    return name, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("assets")
    ap.add_argument("out")
    ap.add_argument("--engine", choices=["mujoco", "physx"], default="mujoco")
    ap.add_argument("--geom", choices=["parts", "union", "sdf"], default="parts")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--shuffle", action="store_true", help="seeded random order, so a partial run is a random subset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(p for p in glob.glob(f"{args.assets}/*.json") if not os.path.basename(p).startswith("_"))
    if args.only:
        paths = [p for p in paths if os.path.basename(p)[:-5] in set(args.only)]
    if args.shuffle:
        np.random.default_rng(0).shuffle(paths)
    jobs = [(p, args.out, args.engine, args.geom, args.repeats) for p in paths]
    with Pool(args.workers, maxtasksperchild=20) as pool:
        results = pool.map(process, jobs, chunksize=1)
    fails = {n: e for n, e in results if e}
    print(f"simulated {len(results) - len(fails)} / {len(results)}")
    for n, e in list(fails.items())[:20]:
        print("  FAIL", n, e)
    prev = json.load(open(f"{args.out}/_fails.json")) if os.path.exists(f"{args.out}/_fails.json") else {}
    prev = {k: v for k, v in prev.items() if k not in {n for n, _ in results}}
    prev.update(fails)
    json.dump(prev, open(f"{args.out}/_fails.json", "w"), indent=1)
