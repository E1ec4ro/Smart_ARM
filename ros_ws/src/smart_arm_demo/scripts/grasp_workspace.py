#!/usr/bin/env python3
"""Shared geometry and limits for virtual Gazebo attach and pick/place heights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import rospy


def param_bool(name: str, default: bool) -> bool:
    v = rospy.get_param(name, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return bool(v)


@dataclass
class GraspWorkspaceConfig:
    cube_size: float
    approach_height: float
    grasp_height_offset: float
    lift_height: float
    place_clearance: float
    place_approach_height: float


@dataclass
class AttachLimits:
    max_distance_m: float
    max_dz_above_cube_center_m: float


def min_cube_center_z_above_table(table_center_z: float, table_size_z: float, cube_size: float) -> float:
    """Minimum world Z of cube center so it sits on the table slab (matches gazebo_attach release clamp)."""
    table_top_z = float(table_center_z) + float(table_size_z) * 0.5
    return table_top_z + float(cube_size) * 0.5 + 0.002


def load_grasp_workspace_config() -> GraspWorkspaceConfig:
    """Load ~ params, apply the same clamps as before (documented in config/workspace.yaml)."""
    _approach_h = float(rospy.get_param("~approach_height", 0.15))
    _place_ah = rospy.get_param("~place_approach_height", None)
    _place_approach_height = float(_place_ah) if _place_ah is not None else min(_approach_h, 0.24)

    cube_size = float(rospy.get_param("~cube_size", rospy.get_param("~target_cube/size", 0.08)))
    grasp_height_offset = float(rospy.get_param("~grasp_height_offset", 0.0))

    _half = cube_size * 0.5
    _min_above_top = float(rospy.get_param("~min_clearance_above_cube_top_m", 0.03))
    _min_grasp_offset = _half + _min_above_top
    if grasp_height_offset < _min_grasp_offset:
        rospy.loginfo(
            "grasp_height_offset %.4f → %.4f (min: half cube + min_clearance_above_cube_top)",
            grasp_height_offset,
            _min_grasp_offset,
        )
        grasp_height_offset = _min_grasp_offset

    approach_height = _approach_h
    _approach_extra = float(rospy.get_param("~min_approach_above_grasp_m", 0.16))
    if approach_height < grasp_height_offset + _approach_extra:
        na = grasp_height_offset + _approach_extra
        rospy.loginfo(
            "approach_height %.4f → %.4f (must stay above grasp pose by min_approach_above_grasp_m)",
            approach_height,
            na,
        )
        approach_height = na

    lift_height = float(rospy.get_param("~lift_height", 0.20))
    if lift_height < grasp_height_offset:
        nh = grasp_height_offset + 0.02
        rospy.logwarn(
            "lift_height %.4f < grasp_height_offset %.4f; using %.4f so lift moves upward",
            lift_height,
            grasp_height_offset,
            nh,
        )
        lift_height = nh

    rospy.loginfo(
        "Heights vs cube center (world): approach +%.3f m, grasp +%.3f m, place_approach +%.3f m",
        approach_height,
        grasp_height_offset,
        _place_approach_height,
    )

    return GraspWorkspaceConfig(
        cube_size=cube_size,
        approach_height=approach_height,
        grasp_height_offset=grasp_height_offset,
        lift_height=lift_height,
        place_clearance=float(rospy.get_param("~place_clearance", 0.01)),
        place_approach_height=_place_approach_height,
    )


def virtual_attach_metrics(
    ee_xyz: Tuple[float, float, float],
    cube_center_base: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Returns (dz_above_cube_center, distance_m) in base_link."""
    ex, ey, ez = float(ee_xyz[0]), float(ee_xyz[1]), float(ee_xyz[2])
    cx, cy, cz = float(cube_center_base[0]), float(cube_center_base[1]), float(cube_center_base[2])
    dz = ez - cz
    d = ((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2) ** 0.5
    return dz, d


def validate_virtual_attach(
    ee_xyz: Tuple[float, float, float],
    cube_center_base: Tuple[float, float, float],
    limits: AttachLimits,
) -> Tuple[float, float]:
    """
    Gazebo /attach teleports the cube to the gripper; refuse if the arm is not near the cube
    (same checks as pick_place_moveit attach step). Returns (dz, distance_m) on success.
    """
    dz, d = virtual_attach_metrics(ee_xyz, cube_center_base)
    if dz > float(limits.max_dz_above_cube_center_m):
        raise RuntimeError(
            f"Refusing attach: EE too high above cube (dz={dz:.3f} m, max "
            f"{float(limits.max_dz_above_cube_center_m):.3f} m). Virtual attach would move cube without contact. "
            "Lower grasp_height_offset or increase attach_max_dz_above_cube_m."
        )
    if d > float(limits.max_distance_m):
        raise RuntimeError(
            f"Refusing to attach: EE too far from cube (dist={d:.3f}m, max={float(limits.max_distance_m):.3f}m)"
        )
    return dz, d
