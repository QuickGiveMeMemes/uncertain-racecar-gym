from __future__ import annotations

import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

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
        self.ghost_vehicle_id = None
        self._joint_names = {}
        self._ghost_joint_names = {}
        self._frame_index = 0
        self._camera_renderer = p.ER_BULLET_HARDWARE_OPENGL
        self._connect()
        self._build_scene()

    def _polyline_normals(self) -> np.ndarray:
        points = self.track.centerline
        previous = np.roll(points, 1, axis=0)
        nxt = np.roll(points, -1, axis=0)
        tangents = nxt - previous
        tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
        return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    def _create_strip_mesh(
        self,
        edge_a: np.ndarray,
        edge_b: np.ndarray,
        z_a: float,
        z_b: float,
        color: list[float],
        closed: bool = True,
    ) -> int:
        vertices = []
        indices = []
        point_count = len(edge_a)
        segment_count = point_count if closed else point_count - 1
        for index in range(point_count):
            vertices.append([float(edge_a[index, 0]), float(edge_a[index, 1]), float(z_a)])
            vertices.append([float(edge_b[index, 0]), float(edge_b[index, 1]), float(z_b)])
        for index in range(segment_count):
            nxt = (index + 1) % point_count
            base = 2 * index
            nxt_base = 2 * nxt
            indices.extend([base, base + 1, nxt_base + 1, base, nxt_base + 1, nxt_base])
        return p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=color,
            specularColor=[0.08, 0.08, 0.08],
            physicsClientId=self.client_id,
        )

    def _create_visual_body(self, visual_shape: int, position: list[float] | None = None) -> None:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=position or [0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

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

        normals = self._polyline_normals()
        centerline = self.track.centerline
        shoulder_outer = 0.5 * self.track.width + 1.3
        road_half_width = 0.5 * self.track.width
        stripe_outer = road_half_width - 0.04
        stripe_inner = road_half_width - 0.24

        left_shoulder_outer = centerline + normals * shoulder_outer
        right_shoulder_outer = centerline - normals * shoulder_outer
        left_edge = centerline + normals * road_half_width
        right_edge = centerline - normals * road_half_width
        left_stripe_inner = centerline + normals * stripe_inner
        right_stripe_inner = centerline - normals * stripe_inner

        shoulder_shape = self._create_strip_mesh(
            left_shoulder_outer,
            right_shoulder_outer,
            z_a=road_height * 0.96,
            z_b=road_height * 0.96,
            color=[0.22, 0.22, 0.24, 1.0],
        )
        self._create_visual_body(shoulder_shape)

        road_shape = self._create_strip_mesh(
            left_edge,
            right_edge,
            z_a=road_height,
            z_b=road_height,
            color=[0.12, 0.12, 0.14, 1.0],
        )
        self._create_visual_body(road_shape)

        left_edge_shape = self._create_strip_mesh(
            centerline + normals * stripe_outer,
            left_stripe_inner,
            z_a=road_height * 1.12,
            z_b=road_height * 1.12,
            color=[0.96, 0.96, 0.96, 1.0],
        )
        right_edge_shape = self._create_strip_mesh(
            right_stripe_inner,
            centerline - normals * stripe_outer,
            z_a=road_height * 1.12,
            z_b=road_height * 1.12,
            color=[0.96, 0.96, 0.96, 1.0],
        )
        self._create_visual_body(left_edge_shape)
        self._create_visual_body(right_edge_shape)

        curb_outer_width = road_half_width + 0.55
        curb_inner_width = road_half_width + 0.12
        left_curb_shape = self._create_strip_mesh(
            centerline + normals * curb_outer_width,
            centerline + normals * curb_inner_width,
            z_a=road_height * 1.02,
            z_b=road_height * 1.02,
            color=[0.78, 0.16, 0.16, 0.92],
        )
        right_curb_shape = self._create_strip_mesh(
            centerline - normals * curb_inner_width,
            centerline - normals * curb_outer_width,
            z_a=road_height * 1.02,
            z_b=road_height * 1.02,
            color=[0.78, 0.16, 0.16, 0.92],
        )
        self._create_visual_body(left_curb_shape)
        self._create_visual_body(right_curb_shape)

        guardrail_inner = centerline + normals * (road_half_width + 0.9)
        guardrail_outer = centerline + normals * (road_half_width + 1.02)
        guardrail_left = self._create_strip_mesh(
            guardrail_inner,
            guardrail_outer,
            z_a=0.30,
            z_b=0.34,
            color=[0.88, 0.9, 0.94, 0.92],
        )
        guardrail_right = self._create_strip_mesh(
            centerline - normals * (road_half_width + 1.02),
            centerline - normals * (road_half_width + 0.9),
            z_a=0.34,
            z_b=0.30,
            color=[0.88, 0.9, 0.94, 0.92],
        )
        self._create_visual_body(guardrail_left)
        self._create_visual_body(guardrail_right)

        center_dash_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.65, 0.045, 0.01],
            rgbaColor=[0.98, 0.86, 0.2, 0.95],
            physicsClientId=self.client_id,
        )
        dash_spacing = 4.8
        dash_length = 1.2
        dash_count = max(1, int(self.track.length / dash_spacing))
        for dash_index in range(dash_count):
            progress = (dash_index * dash_spacing) / max(self.track.length, 1e-9)
            start = self.track.sample(progress)
            end = self.track.sample(progress + dash_length / max(self.track.length, 1e-9))
            midpoint = np.array([(start.x + end.x) * 0.5, (start.y + end.y) * 0.5], dtype=float)
            yaw = math.atan2(end.y - start.y, end.x - start.x)
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=center_dash_shape,
                basePosition=[float(midpoint[0]), float(midpoint[1]), road_height * 1.2],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

        self.vehicle_id, self._joint_names = self._spawn_vehicle(rgba=None)

    def _spawn_vehicle(self, rgba: list[float] | None) -> tuple[int, dict[str, int]]:
        urdf_path = package_asset_path("vehicles/simple_racecar.urdf")
        vehicle_id = p.loadURDF(
            str(urdf_path),
            [0.0, 0.0, 0.2],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        joint_names: dict[str, int] = {}
        if rgba is not None:
            p.changeVisualShape(vehicle_id, -1, rgbaColor=rgba, physicsClientId=self.client_id)
        p.setCollisionFilterGroupMask(vehicle_id, -1, 0, 0, physicsClientId=self.client_id)
        for joint_index in range(p.getNumJoints(vehicle_id, physicsClientId=self.client_id)):
            joint_name = p.getJointInfo(vehicle_id, joint_index, physicsClientId=self.client_id)[1].decode("utf-8")
            joint_names[joint_name] = joint_index
            if rgba is not None:
                p.changeVisualShape(vehicle_id, joint_index, rgbaColor=rgba, physicsClientId=self.client_id)
            p.setCollisionFilterGroupMask(vehicle_id, joint_index, 0, 0, physicsClientId=self.client_id)
        return vehicle_id, joint_names

    def _ensure_ghost_vehicle(self) -> None:
        if self.ghost_vehicle_id is None:
            self.ghost_vehicle_id, self._ghost_joint_names = self._spawn_vehicle([0.32, 0.72, 1.0, 0.38])

    def _apply_vehicle_state(self, vehicle_id: int, joint_names: dict[str, int], render_state: dict) -> None:
        position = [render_state["x"], render_state["y"], 0.22]
        orientation = p.getQuaternionFromEuler([0.0, 0.0, render_state["yaw"]])
        p.resetBasePositionAndOrientation(vehicle_id, position, orientation, physicsClientId=self.client_id)

        steer_angle = render_state["steering_angle"]
        wheel_rotation = render_state["wheel_rotation"]
        front_steer_joints = ("front_left_steer", "front_right_steer")
        wheel_joints = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
        for name in front_steer_joints:
            p.resetJointState(vehicle_id, joint_names[name], targetValue=steer_angle, physicsClientId=self.client_id)
        for name in wheel_joints:
            p.resetJointState(vehicle_id, joint_names[name], targetValue=wheel_rotation, physicsClientId=self.client_id)
        self._frame_index = int(render_state.get("frame_index", self._frame_index + 1))

    def update(self, render_state: dict, comparison_state: dict | None = None) -> None:
        self._apply_vehicle_state(self.vehicle_id, self._joint_names, render_state)
        if comparison_state is not None:
            self._ensure_ghost_vehicle()
            self._apply_vehicle_state(self.ghost_vehicle_id, self._ghost_joint_names, comparison_state)
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

    @staticmethod
    def _matrix4(values: list[float] | tuple[float, ...]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape((4, 4), order="F")

    def _project_xy(self, xy_points: np.ndarray, view: list[float] | tuple[float, ...], projection: list[float] | tuple[float, ...]) -> np.ndarray:
        if xy_points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        view_matrix = self._matrix4(view)
        projection_matrix = self._matrix4(projection)
        points = np.asarray(xy_points, dtype=np.float32)
        homogeneous = np.concatenate(
            [
                points,
                np.full((points.shape[0], 1), 0.28, dtype=np.float32),
                np.ones((points.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        clip = (projection_matrix @ view_matrix @ homogeneous.T).T
        w = np.maximum(clip[:, 3:4], 1e-6)
        ndc = clip[:, :3] / w
        pixels = np.empty((points.shape[0], 2), dtype=np.float32)
        pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * float(self.width - 1)
        pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(self.height - 1)
        valid = np.logical_and.reduce(
            [
                clip[:, 3] > 1e-6,
                ndc[:, 2] > -1.5,
                ndc[:, 2] < 1.5,
            ]
        )
        pixels[~valid] = np.nan
        return pixels

    def _overlay_planner_debug(
        self,
        frame: np.ndarray,
        planner_debug: dict | None,
        view: list[float] | tuple[float, ...],
        projection: list[float] | tuple[float, ...],
    ) -> np.ndarray:
        if planner_debug is None:
            return frame
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        candidate_xy = np.asarray(planner_debug.get("candidate_xy", np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        final_xy = np.asarray(planner_debug.get("final_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)

        for trajectory in candidate_xy:
            pixels = self._project_xy(trajectory, view, projection)
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            if len(pixels) >= 2:
                draw.line([tuple(point) for point in pixels], fill=(84, 180, 255, 46), width=1)

        if len(final_xy) >= 2:
            pixels = self._project_xy(final_xy, view, projection)
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            if len(pixels) >= 2:
                draw.line([tuple(point) for point in pixels], fill=(255, 184, 44, 255), width=4)
                draw.line([tuple(point) for point in pixels], fill=(255, 236, 180, 176), width=2)

        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def render(
        self,
        render_state: dict,
        comparison_state: dict | None = None,
        planner_debug: dict | None = None,
    ) -> np.ndarray | None:
        mode = self.render_mode.replace("rgb_array_", "")
        if self.render_mode == "human" and p.getConnectionInfo(self.client_id)["connectionMethod"] == p.GUI:
            self.update(render_state, comparison_state=comparison_state)
            return None

        self.update(render_state, comparison_state=comparison_state)
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
        frame = np.reshape(rgb, (self.height, self.width, 4))[:, :, :3]
        return self._overlay_planner_debug(frame, planner_debug, view, projection)

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
