#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import LinkStates, ModelStates
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from sensor_msgs.msg import Image


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


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    q = Quaternion()
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    return q


def quat_conjugate(q: Quaternion) -> Quaternion:
    qc = Quaternion()
    qc.w = q.w
    qc.x = -q.x
    qc.y = -q.y
    qc.z = -q.z
    return qc


def quat_rotate(q: Quaternion, v: np.ndarray) -> np.ndarray:
    vq = Quaternion(x=float(v[0]), y=float(v[1]), z=float(v[2]), w=0.0)
    q_inv = quat_conjugate(q)
    out = quat_multiply(quat_multiply(q, vq), q_inv)
    return np.array([out.x, out.y, out.z], dtype=np.float64)


@dataclass
class CameraConfig:
    name: str
    image_topic: str
    image_width: int
    image_height: int
    hfov: float
    camera_pos_w: np.ndarray
    camera_q_w: Quaternion


@dataclass
class Detection:
    xyz: np.ndarray
    score: float
    stamp: rospy.Time


class NeuralNetworkCamera:
    def __init__(self) -> None:
        rospy.init_node("neural_network_camera", anonymous=True)
        self.bridge = CvBridge()

        self.image_pub = rospy.Publisher("/overhead_camera/image_detected", Image, queue_size=1)
        self.cube_position_pub = rospy.Publisher("/detected_cube_position", Point, queue_size=1)
        self.cube_pose_pub = rospy.Publisher("/detected_cube_pose_world", PoseStamped, queue_size=1)
        self.goal_pose_pub = rospy.Publisher("/detected_goal_pose_world", PoseStamped, queue_size=1)

        self.use_gazebo_ground_truth = param_bool("~use_gazebo_ground_truth", True)
        self.ground_truth_only = param_bool("~ground_truth_only", False)
        self.fallback_to_ground_truth = param_bool("~fallback_to_ground_truth", True)

        self.cube_model_name = str(rospy.get_param("~cube_model_name", "target_cube"))
        self.goal_model_name = str(rospy.get_param("~goal_model_name", "cube_container"))
        self.cube_link_name = str(rospy.get_param("~cube_link_name", "target_cube::cube_link"))
        self.goal_link_name = str(rospy.get_param("~goal_link_name", "cube_container::container_link"))

        self.fusion_max_age_s = float(rospy.get_param("~fusion_max_age_s", 0.45))
        self.min_cameras_for_fusion = int(rospy.get_param("~min_cameras_for_fusion", 1))
        self.min_cube_score_publish = float(rospy.get_param("~min_cube_score_publish", 0.15))
        self.min_goal_score_publish = float(rospy.get_param("~min_goal_score_publish", 0.12))
        self.ema_alpha = float(rospy.get_param("~pose_ema_alpha", 0.35))

        self._ema_cube: Optional[np.ndarray] = None
        self._ema_goal: Optional[np.ndarray] = None

        self._model_states = None
        self._link_states = None
        if self.use_gazebo_ground_truth:
            rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_model_states, queue_size=1)
            rospy.Subscriber("/gazebo/link_states", LinkStates, self._on_link_states, queue_size=1)

        self.lower_red1 = np.array([0, 80, 60])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 80, 60])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([95, 255, 255])

        self.world_frame = str(rospy.get_param("~world_frame", "world"))
        self.table_center_z = float(rospy.get_param("~table_center_z", rospy.get_param("~table/center_z", 0.40)))
        self.table_size_z = float(rospy.get_param("~table_size_z", rospy.get_param("~table/size_z", 0.05)))
        self.container_center_z = float(rospy.get_param("~container_center_z", rospy.get_param("~goal_platform/center_z", 0.46)))
        self.container_size_z = float(rospy.get_param("~container_size_z", rospy.get_param("~goal_platform/size_z", 0.06)))
        self.cube_size = float(rospy.get_param("~cube_size", rospy.get_param("~target_cube/size", 0.08)))
        self.goal_pose_keep_link_z = param_bool("~goal_pose_keep_link_z", True)

        self.cameras = self._load_camera_configs()
        self.camera_debug_pubs: Dict[str, rospy.Publisher] = {}
        self._subs: List[rospy.Subscriber] = []

        self._cube_dets: Dict[str, Detection] = {}
        self._goal_dets: Dict[str, Detection] = {}

        for idx, cam in enumerate(self.cameras):
            dbg_topic = f"/overhead_camera/{cam.name}/image_detected"
            self.camera_debug_pubs[cam.name] = rospy.Publisher(dbg_topic, Image, queue_size=1)
            sub = rospy.Subscriber(cam.image_topic, Image, self._make_image_cb(cam, idx == 0), queue_size=1)
            self._subs.append(sub)
            rospy.loginfo("camera[%s]: topic=%s debug=%s", cam.name, cam.image_topic, dbg_topic)

        rospy.loginfo(
            "Vision AI initialized: cameras=%d fusion(min=%d age=%.2fs) gt_only=%s fallback_gt=%s",
            len(self.cameras),
            self.min_cameras_for_fusion,
            self.fusion_max_age_s,
            self.ground_truth_only,
            self.fallback_to_ground_truth,
        )

    def _load_camera_configs(self) -> List[CameraConfig]:
        cameras_raw = rospy.get_param("~camera_sources", None)
        cams: List[CameraConfig] = []

        if isinstance(cameras_raw, list) and cameras_raw:
            for i, item in enumerate(cameras_raw):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", f"cam{i}"))
                image_topic = str(item.get("image_topic", "/overhead_camera/image_raw"))
                w = int(item.get("image_width", rospy.get_param("~image_width", 800)))
                h = int(item.get("image_height", rospy.get_param("~image_height", 600)))
                hfov = float(item.get("hfov", rospy.get_param("~hfov", 1.6)))
                xyz = item.get("camera_xyz", rospy.get_param("~camera_xyz", [0.619308, 0.874419, 1.18362]))
                rpy = item.get("camera_rpy", rospy.get_param("~camera_rpy", [0.027906, 0.947704, -3.09384]))
                cams.append(
                    CameraConfig(
                        name=name,
                        image_topic=image_topic,
                        image_width=w,
                        image_height=h,
                        hfov=hfov,
                        camera_pos_w=np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64),
                        camera_q_w=quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])),
                    )
                )

        if not cams:
            xyz = rospy.get_param("~camera_xyz", [0.619308, 0.874419, 1.18362])
            rpy = rospy.get_param("~camera_rpy", [0.027906, 0.947704, -3.09384])
            cams.append(
                CameraConfig(
                    name="main",
                    image_topic=str(rospy.get_param("~image_topic", "/overhead_camera/image_raw")),
                    image_width=int(rospy.get_param("~image_width", 800)),
                    image_height=int(rospy.get_param("~image_height", 600)),
                    hfov=float(rospy.get_param("~hfov", 1.6)),
                    camera_pos_w=np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64),
                    camera_q_w=quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])),
                )
            )
        return cams

    def _on_model_states(self, msg: ModelStates) -> None:
        self._model_states = msg

    def _on_link_states(self, msg: LinkStates) -> None:
        self._link_states = msg

    def _get_model_pose_world(self, model_name: str) -> Optional[PoseStamped]:
        if self._model_states is None:
            return None
        try:
            idx = self._model_states.name.index(model_name)
        except ValueError:
            return None
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.world_frame
        pose.pose = self._model_states.pose[idx]
        return pose

    def _get_link_pose_world(self, link_name: str) -> Optional[PoseStamped]:
        if self._link_states is None:
            return None
        try:
            idx = self._link_states.name.index(link_name)
        except ValueError:
            return None
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.world_frame
        pose.pose = self._link_states.pose[idx]
        return pose

    def _camera_intrinsics(self, cam: CameraConfig, w: int, h: int) -> Tuple[float, float, float, float]:
        cx = (float(w) - 1.0) * 0.5
        cy = (float(h) - 1.0) * 0.5
        fx = (float(cam.image_width) * 0.5) / math.tan(cam.hfov * 0.5)
        vfov = 2.0 * math.atan(math.tan(cam.hfov * 0.5) * (float(cam.image_height) / max(float(cam.image_width), 1.0)))
        fy = (float(cam.image_height) * 0.5) / math.tan(vfov * 0.5)
        return fx, fy, cx, cy

    def _pixel_ray_world(self, cam: CameraConfig, u: float, v: float, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        fx, fy, cx, cy = self._camera_intrinsics(cam, w, h)
        x = (u - cx) / fx
        y = (v - cy) / fy
        d_cam = np.array([x, y, 1.0], dtype=np.float64)
        d_cam /= max(np.linalg.norm(d_cam), 1e-9)
        d_w = quat_rotate(cam.camera_q_w, d_cam)
        return cam.camera_pos_w.copy(), d_w

    def _intersect_plane_z(self, cam: CameraConfig, u: float, v: float, plane_z: float, w: int, h: int) -> Optional[np.ndarray]:
        o, d = self._pixel_ray_world(cam, u, v, w, h)
        if abs(float(d[2])) < 1e-8:
            return None
        t = (float(plane_z) - float(o[2])) / float(d[2])
        if t <= 0.0:
            return None
        return o + t * d

    def detect_red_cube(self, cv_image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox, best_area = None, 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 120.0:
                continue
            if area > best_area:
                best_area = area
                best_bbox = cv2.boundingRect(contour)

        h_img, w_img = cv_image.shape[:2]
        score = 0.0 if best_bbox is None else min(best_area / max(float(w_img * h_img), 1.0) * 60.0, 1.0)
        return best_bbox, score

    def detect_green_goal(self, cv_image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox, best_area = None, 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 350.0:
                continue
            if area > best_area:
                best_area = area
                best_bbox = cv2.boundingRect(contour)

        h_img, w_img = cv_image.shape[:2]
        score = 0.0 if best_bbox is None else min(best_area / max(float(w_img * h_img), 1.0) * 15.0, 1.0)
        return best_bbox, score

    def _draw_bbox(self, img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], text: str) -> None:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, 0.6, 2)
        y_top = max(y - th - baseline - 4, 0)
        y_text = max(y - baseline - 2, th + baseline + 2)
        cv2.rectangle(img, (x, y_top), (x + tw + 6, y), color, -1)
        cv2.putText(img, text, (x + 3, y_text), font, 0.6, (255, 255, 255), 2)

    def _publish_debug(self, topic_pub: rospy.Publisher, header, image: np.ndarray) -> None:
        try:
            out = self.bridge.cv2_to_imgmsg(image, "bgr8")
            out.header = header
            topic_pub.publish(out)
        except CvBridgeError as e:
            rospy.logerr("cv_bridge publish error: %s", e)

    def _update_ema(self, old_val: Optional[np.ndarray], new_val: np.ndarray) -> np.ndarray:
        if old_val is None:
            return new_val.copy()
        a = float(np.clip(self.ema_alpha, 0.01, 1.0))
        return old_val * (1.0 - a) + new_val * a

    def _fuse_detections(self, dets: Dict[str, Detection], now: rospy.Time) -> Tuple[Optional[np.ndarray], float, int]:
        valid: List[Detection] = []
        for det in dets.values():
            age = max((now - det.stamp).to_sec(), 0.0)
            if age <= self.fusion_max_age_s:
                valid.append(det)
        if len(valid) < self.min_cameras_for_fusion:
            return None, 0.0, len(valid)

        w_sum = 0.0
        pos = np.zeros(3, dtype=np.float64)
        for det in valid:
            w = max(float(det.score), 1e-3)
            w_sum += w
            pos += det.xyz * w
        if w_sum <= 1e-6:
            return None, 0.0, len(valid)
        pos /= w_sum
        conf = float(sum([d.score for d in valid]) / max(len(valid), 1))
        return pos, conf, len(valid)

    def _publish_cube_pose(self, xyz: np.ndarray, stamp: rospy.Time) -> None:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.world_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.w = 1.0
        self.cube_pose_pub.publish(pose)

    def _publish_goal_pose(self, xyz: np.ndarray, stamp: rospy.Time) -> None:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.world_frame
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.w = 1.0
        self.goal_pose_pub.publish(pose)

    def _publish_ground_truth_if_available(self, stamp: rospy.Time) -> None:
        table_top_z = self.table_center_z + self.table_size_z * 0.5
        container_top_z = self.container_center_z + self.container_size_z * 0.5

        gt_cube = self._get_link_pose_world(self.cube_link_name) or self._get_model_pose_world(self.cube_model_name)
        if gt_cube is not None:
            gt_cube.header.stamp = stamp
            gt_cube.pose.position.z = float(table_top_z + self.cube_size * 0.5)
            self.cube_pose_pub.publish(gt_cube)

        gt_goal = self._get_link_pose_world(self.goal_link_name) or self._get_model_pose_world(self.goal_model_name)
        if gt_goal is not None:
            gt_goal.header.stamp = stamp
            if not self.goal_pose_keep_link_z:
                gt_goal.pose.position.z = float(container_top_z)
            self.goal_pose_pub.publish(gt_goal)

    def _make_image_cb(self, cam: CameraConfig, is_primary: bool):
        def _cb(msg: Image) -> None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr("cv_bridge conversion error (%s): %s", cam.name, e)
                return

            cube_bbox, cube_score = self.detect_red_cube(cv_image)
            goal_bbox, goal_score = self.detect_green_goal(cv_image)
            h_img, w_img = cv_image.shape[:2]

            table_top_z = self.table_center_z + self.table_size_z * 0.5
            container_top_z = self.container_center_z + self.container_size_z * 0.5

            if cube_bbox is not None:
                self._draw_bbox(cv_image, cube_bbox, (20, 40, 255), f"cube {cube_score:.2f}")
                x, y, w, h = cube_bbox
                u = x + w * 0.5
                v = y + h * 0.5
                p = Point(x=u, y=v, z=cube_score)
                self.cube_position_pub.publish(p)
                if not self.ground_truth_only:
                    hit = self._intersect_plane_z(cam, u, v, table_top_z, w_img, h_img)
                    if hit is not None:
                        hit[2] = table_top_z + self.cube_size * 0.5
                        self._cube_dets[cam.name] = Detection(xyz=hit, score=cube_score, stamp=msg.header.stamp)

            if goal_bbox is not None:
                self._draw_bbox(cv_image, goal_bbox, (0, 200, 0), f"goal {goal_score:.2f}")
                gx, gy, gw, gh = goal_bbox
                u = gx + gw * 0.5
                v = gy + gh * 0.5
                if not self.ground_truth_only:
                    hit = self._intersect_plane_z(cam, u, v, container_top_z, w_img, h_img)
                    if hit is not None:
                        if self.goal_pose_keep_link_z:
                            hit[2] = container_top_z
                        self._goal_dets[cam.name] = Detection(xyz=hit, score=goal_score, stamp=msg.header.stamp)

            cv2.putText(
                cv_image,
                f"{cam.name} topic={cam.image_topic}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            self._publish_debug(self.camera_debug_pubs[cam.name], msg.header, cv_image)
            if is_primary:
                self._publish_debug(self.image_pub, msg.header, cv_image)

            self._publish_fused(msg.header.stamp)

        return _cb

    def _publish_fused(self, stamp: rospy.Time) -> None:
        if self.use_gazebo_ground_truth and self.ground_truth_only:
            self._publish_ground_truth_if_available(stamp)
            return

        cube_xyz, cube_conf, cube_n = self._fuse_detections(self._cube_dets, stamp)
        goal_xyz, goal_conf, goal_n = self._fuse_detections(self._goal_dets, stamp)

        if cube_xyz is not None and cube_conf >= self.min_cube_score_publish:
            self._ema_cube = self._update_ema(self._ema_cube, cube_xyz)
            self._publish_cube_pose(self._ema_cube, stamp)
        elif self.use_gazebo_ground_truth and self.fallback_to_ground_truth:
            gt_cube = self._get_link_pose_world(self.cube_link_name) or self._get_model_pose_world(self.cube_model_name)
            if gt_cube is not None:
                table_top_z = self.table_center_z + self.table_size_z * 0.5
                gt_cube.header.stamp = stamp
                gt_cube.pose.position.z = float(table_top_z + self.cube_size * 0.5)
                self.cube_pose_pub.publish(gt_cube)

        if goal_xyz is not None and goal_conf >= self.min_goal_score_publish:
            self._ema_goal = self._update_ema(self._ema_goal, goal_xyz)
            self._publish_goal_pose(self._ema_goal, stamp)
        elif self.use_gazebo_ground_truth and self.fallback_to_ground_truth:
            gt_goal = self._get_link_pose_world(self.goal_link_name) or self._get_model_pose_world(self.goal_model_name)
            if gt_goal is not None:
                gt_goal.header.stamp = stamp
                if not self.goal_pose_keep_link_z:
                    container_top_z = self.container_center_z + self.container_size_z * 0.5
                    gt_goal.pose.position.z = float(container_top_z)
                self.goal_pose_pub.publish(gt_goal)

        rospy.logdebug(
            "fusion cube: n=%d conf=%.3f goal: n=%d conf=%.3f",
            cube_n,
            cube_conf,
            goal_n,
            goal_conf,
        )


def main() -> None:
    try:
        NeuralNetworkCamera()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
