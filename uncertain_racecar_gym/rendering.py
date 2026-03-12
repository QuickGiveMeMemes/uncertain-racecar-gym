from __future__ import annotations

import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p

from uncertain_racecar_gym.common import package_asset_path
from uncertain_racecar_gym.scenario import Scenario
from uncertain_racecar_gym.track import TrackModel


class PyBulletMirrorRenderer:
    def __init__(self, scenario: Scenario, track: TrackModel, render_mode: str, width: int = 1280, height: int = 720):
        self.scenario = scenario
        self.track = track
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.client_id = None
        self.vehicle_id = None
        self._joint_names = {}
        self._frame_index = 0
        self._camera_renderer = p.ER_BULLET_HARDWARE_OPENGL
        self._connect()
        self._build_scene()

    def _connect(self) -> None:
        use_gui = self.render_mode == "human" and (os.environ.get("DISPLAY") or os.name == "nt" or os.uname().sysname == "Darwin")
        connection_mode = p.GUI if use_gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)

    def _build_scene(self) -> None:
        road_height = 0.02
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[120.0, 120.0, 0.02],
                physicsClientId=self.client_id,
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[120.0, 120.0, 0.02],
                rgbaColor=[0.2, 0.37, 0.2, 1.0],
                physicsClientId=self.client_id,
            ),
            basePosition=[0.0, 0.0, -0.02],
            physicsClientId=self.client_id,
        )

        road_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, self.track.width * 0.5, road_height],
            rgbaColor=[0.16, 0.16, 0.18, 1.0],
            physicsClientId=self.client_id,
        )
        edge_strip_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.08, 0.01],
            rgbaColor=[0.95, 0.95, 0.95, 1.0],
            physicsClientId=self.client_id,
        )
        center_dash_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.28, 0.03, 0.01],
            rgbaColor=[0.98, 0.86, 0.2, 1.0],
            physicsClientId=self.client_id,
        )
        wall_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.08, 0.25],
            rgbaColor=[0.82, 0.16, 0.16, 1.0],
            physicsClientId=self.client_id,
        )

        for index, segment in enumerate(self.track._segments):
            length = self.track._segment_lengths[index]
            if length < 1e-6:
                continue
            midpoint = self.track.centerline[index] + 0.5 * segment
            yaw = math.atan2(segment[1], segment[0])
            road_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length * 0.5, self.track.width * 0.5, road_height], physicsClientId=self.client_id)
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=road_collision,
                baseVisualShapeIndex=road_shape,
                basePosition=[midpoint[0], midpoint[1], road_height],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

            tangent = segment / length
            normal = np.array([-tangent[1], tangent[0]])
            edge_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length * 0.5, 0.08, 0.01],
                physicsClientId=self.client_id,
            )
            for edge_sign in (-1.0, 1.0):
                edge_pos = midpoint + normal * edge_sign * (self.track.width * 0.5 - 0.08)
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=edge_collision,
                    baseVisualShapeIndex=edge_strip_shape,
                    basePosition=[edge_pos[0], edge_pos[1], road_height * 2.2],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )

            dash_count = max(1, int(length / 1.5))
            dash_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.28, 0.03, 0.01],
                physicsClientId=self.client_id,
            )
            for dash_index in range(dash_count):
                alpha = (dash_index + 0.5) / dash_count
                dash_pos = self.track.centerline[index] + segment * alpha
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=dash_collision,
                    baseVisualShapeIndex=center_dash_shape,
                    basePosition=[dash_pos[0], dash_pos[1], road_height * 2.4],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )

            for wall_sign in (-1.0, 1.0):
                wall_pos = midpoint + normal * wall_sign * (self.track.width * 0.5)
                wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length * 0.5, 0.08, 0.25], physicsClientId=self.client_id)
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=wall_collision,
                    baseVisualShapeIndex=wall_shape,
                    basePosition=[wall_pos[0], wall_pos[1], 0.25],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )

        urdf_path = package_asset_path("vehicles/simple_racecar.urdf")
        self.vehicle_id = p.loadURDF(str(urdf_path), [0.0, 0.0, 0.2], useFixedBase=False, physicsClientId=self.client_id)
        for joint_index in range(p.getNumJoints(self.vehicle_id, physicsClientId=self.client_id)):
            joint_name = p.getJointInfo(self.vehicle_id, joint_index, physicsClientId=self.client_id)[1].decode("utf-8")
            self._joint_names[joint_name] = joint_index
        p.changeDynamics(self.vehicle_id, -1, mass=1.0, linearDamping=0.0, angularDamping=0.0, physicsClientId=self.client_id)

    def update(self, render_state: dict) -> None:
        position = [render_state["x"], render_state["y"], 0.22]
        orientation = p.getQuaternionFromEuler([0.0, 0.0, render_state["yaw"]])
        p.resetBasePositionAndOrientation(self.vehicle_id, position, orientation, physicsClientId=self.client_id)

        steer_angle = render_state["steering_angle"]
        wheel_rotation = render_state["wheel_rotation"]
        front_steer_joints = ("front_left_steer", "front_right_steer")
        wheel_joints = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
        for name in front_steer_joints:
            p.resetJointState(self.vehicle_id, self._joint_names[name], targetValue=steer_angle, physicsClientId=self.client_id)
        for name in wheel_joints:
            p.resetJointState(self.vehicle_id, self._joint_names[name], targetValue=wheel_rotation, physicsClientId=self.client_id)
        self._frame_index = int(render_state.get("frame_index", self._frame_index + 1))
        p.stepSimulation(physicsClientId=self.client_id)

    def _camera(self, render_state: dict, mode: str):
        x = render_state["x"]
        y = render_state["y"]
        yaw = render_state["yaw"]

        if mode == "birds_eye":
            target = [x, y, 0.0]
            view = p.computeViewMatrixFromYawPitchRoll(target, distance=18.0, yaw=0.0, pitch=-89.0, roll=0.0, upAxisIndex=2)
        elif mode == "cinematic":
            orbit_yaw = math.degrees((self._frame_index * 0.03) % (2.0 * math.pi))
            target = [x, y, 0.0]
            view = p.computeViewMatrixFromYawPitchRoll(target, distance=7.5, yaw=orbit_yaw, pitch=-20.0, roll=0.0, upAxisIndex=2)
        else:
            camera_position = np.array([x, y, 0.45]) + np.array([-4.5 * math.cos(yaw), -4.5 * math.sin(yaw), 1.6])
            target = np.array([x, y, 0.35]) + np.array([2.5 * math.cos(yaw), 2.5 * math.sin(yaw), 0.2])
            view = p.computeViewMatrix(camera_position, target, [0.0, 0.0, 1.0])

        projection = p.computeProjectionMatrixFOV(fov=60.0, aspect=float(self.width) / float(self.height), nearVal=0.05, farVal=100.0)
        return view, projection

    def render(self, render_state: dict) -> np.ndarray | None:
        mode = self.render_mode.replace("rgb_array_", "")
        if self.render_mode == "human" and p.getConnectionInfo(self.client_id)["connectionMethod"] == p.GUI:
            self.update(render_state)
            return None

        self.update(render_state)
        view, projection = self._camera(render_state, mode if mode in {"follow", "birds_eye", "cinematic"} else "follow")
        try:
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                renderer=self._camera_renderer,
                viewMatrix=view,
                projectionMatrix=projection,
                shadow=1,
                lightDirection=[1.2, -0.8, 2.6],
                lightColor=[1.0, 0.98, 0.95],
                lightAmbientCoeff=0.65,
                lightDiffuseCoeff=0.55,
                lightSpecularCoeff=0.15,
                physicsClientId=self.client_id,
            )
        except Exception:
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                renderer=p.ER_TINY_RENDERER,
                viewMatrix=view,
                projectionMatrix=projection,
                physicsClientId=self.client_id,
            )
        return np.reshape(rgb, (self.height, self.width, 4))[:, :, :3]

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None


def write_video(frames: list[np.ndarray], output_path: str | Path, fps: int) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame.astype(np.uint8))
    return path
