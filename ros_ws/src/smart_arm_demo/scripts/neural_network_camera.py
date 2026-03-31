#!/usr/bin/env python3
import math
import cv2
import rospy
import numpy as np
from typing import Optional, Tuple

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from gazebo_msgs.msg import ModelStates, LinkStates


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


class NeuralNetworkCamera:
    def __init__(self) -> None:
        rospy.init_node("neural_network_camera", anonymous=True)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/overhead_camera/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/overhead_camera/image_detected", Image, queue_size=1)
        self.cube_position_pub = rospy.Publisher("/detected_cube_position", Point, queue_size=1)
        self.cube_pose_pub = rospy.Publisher("/detected_cube_pose_world", PoseStamped, queue_size=1)
        self.goal_pose_pub = rospy.Publisher("/detected_goal_pose_world", PoseStamped, queue_size=1)

        self.use_gazebo_ground_truth = param_bool("~use_gazebo_ground_truth", True)
        self.ground_truth_only = param_bool("~ground_truth_only", True)
        self.cube_model_name = str(rospy.get_param("~cube_model_name", "target_cube"))
        self.goal_model_name = str(rospy.get_param("~goal_model_name", "cube_container"))
        self.cube_link_name = str(rospy.get_param("~cube_link_name", "target_cube::cube_link"))
        self.goal_link_name = str(rospy.get_param("~goal_link_name", "cube_container::container_link"))
        self._model_states = None
        self._link_states = None
        if self.use_gazebo_ground_truth:
            rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_model_states, queue_size=1)
            rospy.Subscriber("/gazebo/link_states", LinkStates, self._on_link_states, queue_size=1)
        self._last_gt_log = rospy.Time(0)

        self.lower_red1 = np.array([0, 80, 60])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 80, 60])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([95, 255, 255])

        self.image_width = int(rospy.get_param("~image_width", 800))
        self.image_height = int(rospy.get_param("~image_height", 600))
        self.hfov = float(rospy.get_param("~hfov", 1.6))

        self.world_frame = str(rospy.get_param("~world_frame", "world"))
        cam_xyz = rospy.get_param("~camera_xyz", [0.619308, 0.874419, 1.18362])
        cam_rpy = rospy.get_param("~camera_rpy", [0.027906, 0.947704, -3.09384])

        self.camera_pos_w = np.array([float(cam_xyz[0]), float(cam_xyz[1]), float(cam_xyz[2])], dtype=np.float64)
        self.camera_q_w = quat_from_rpy(float(cam_rpy[0]), float(cam_rpy[1]), float(cam_rpy[2]))

        self.table_center_z = float(rospy.get_param("~table_center_z", rospy.get_param("~table/center_z", 0.40)))
        self.table_size_z = float(rospy.get_param("~table_size_z", rospy.get_param("~table/size_z", 0.05)))
        self.container_center_z = float(
            rospy.get_param("~container_center_z", rospy.get_param("~goal_platform/center_z", 0.46))
        )
        self.container_size_z = float(rospy.get_param("~container_size_z", rospy.get_param("~goal_platform/size_z", 0.06)))

        self.cube_size = float(rospy.get_param("~cube_size", rospy.get_param("~target_cube/size", 0.08)))

        rospy.loginfo("Vision estimator initialized: red cube + green goal, pixel->world via ray-plane.")

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

    def _camera_intrinsics(self, w: int, h: int) -> Tuple[float, float, float, float]:
        cx = (float(w) - 1.0) * 0.5
        cy = (float(h) - 1.0) * 0.5
        fx = (float(w) * 0.5) / math.tan(self.hfov * 0.5)
        vfov = 2.0 * math.atan(math.tan(self.hfov * 0.5) * (float(h) / max(float(w), 1.0)))
        fy = (float(h) * 0.5) / math.tan(vfov * 0.5)
        return fx, fy, cx, cy

    def _pixel_ray_world(self, u: float, v: float, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        fx, fy, cx, cy = self._camera_intrinsics(w, h)
        x = (u - cx) / fx
        y = (v - cy) / fy
        d_cam = np.array([x, y, 1.0], dtype=np.float64)
        d_cam /= max(np.linalg.norm(d_cam), 1e-9)
        d_w = quat_rotate(self.camera_q_w, d_cam)
        return self.camera_pos_w.copy(), d_w

    def _intersect_plane_z(self, u: float, v: float, plane_z: float, w: int, h: int) -> Optional[np.ndarray]:
        o, d = self._pixel_ray_world(u, v, w, h)
        if abs(float(d[2])) < 1e-8:
            return None
        t = (float(plane_z) - float(o[2])) / float(d[2])
        if t <= 0.0:
            return None
        p = o + t * d
        return p

    def detect_red_cube(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_area = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 150.0:
                continue
            if area > best_area:
                best_area = area
                best_bbox = cv2.boundingRect(contour)

        h_img, w_img = cv_image.shape[:2]
        score = 0.0 if best_bbox is None else min(best_area / max(float(w_img * h_img), 1.0) * 50.0, 1.0)
        return best_bbox, score

    def detect_green_goal(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_bbox = None
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 500.0:
                continue
            if area > best_area:
                best_area = area
                best_bbox = cv2.boundingRect(contour)

        h_img, w_img = cv_image.shape[:2]
        score = 0.0 if best_bbox is None else min(best_area / max(float(w_img * h_img), 1.0) * 10.0, 1.0)
        return best_bbox, score

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge conversion error: {e}")
            return

        bbox, score = self.detect_red_cube(cv_image)
        goal_bbox, goal_score = self.detect_green_goal(cv_image)

        h_img, w_img = cv_image.shape[:2]
        w_eff = int(self.image_width) if int(self.image_width) > 0 else int(w_img)
        h_eff = int(self.image_height) if int(self.image_height) > 0 else int(h_img)

        if self.use_gazebo_ground_truth and self.ground_truth_only:
            table_top_z = self.table_center_z + self.table_size_z * 0.5
            container_top_z = self.container_center_z + self.container_size_z * 0.5

            gt_cube = self._get_link_pose_world(self.cube_link_name) or self._get_model_pose_world(self.cube_model_name)
            if gt_cube is not None:
                gt_cube.pose.position.z = float(table_top_z + self.cube_size * 0.5)
                self.cube_pose_pub.publish(gt_cube)

            gt_goal = self._get_link_pose_world(self.goal_link_name) or self._get_model_pose_world(self.goal_model_name)
            if gt_goal is not None:
                gt_goal.pose.position.z = float(container_top_z)
                self.goal_pose_pub.publish(gt_goal)

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

            label = f"Red cube {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            y_top = max(y - th - baseline - 6, 0)
            y_text = max(y - baseline - 3, th + baseline + 3)
            cv2.rectangle(cv_image, (x, y_top), (x + tw + 6, y), (255, 0, 0), -1)
            cv2.putText(cv_image, label, (x + 3, y_text), font, font_scale, (255, 255, 255), thickness)

            p = Point()
            p.x = x + w / 2.0
            p.y = y + h / 2.0
            p.z = score
            self.cube_position_pub.publish(p)

            table_top_z = self.table_center_z + self.table_size_z * 0.5
            hit = None if (self.use_gazebo_ground_truth and self.ground_truth_only) else self._intersect_plane_z(p.x, p.y, table_top_z, w_eff, h_eff)
            if hit is not None:
                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self.world_frame
                pose.pose.position.x = float(hit[0])
                pose.pose.position.y = float(hit[1])
                pose.pose.position.z = float(table_top_z + self.cube_size * 0.5)
                pose.pose.orientation.w = 1.0
                self.cube_pose_pub.publish(pose)
            elif self.use_gazebo_ground_truth:
                gt = self._get_link_pose_world(self.cube_link_name) or self._get_model_pose_world(self.cube_model_name)
                if gt is not None:
                    gt.pose.position.z = float(table_top_z + self.cube_size * 0.5)
                    self.cube_pose_pub.publish(gt)
        elif self.use_gazebo_ground_truth:
            gt = self._get_link_pose_world(self.cube_link_name) or self._get_model_pose_world(self.cube_model_name)
            if gt is not None:
                table_top_z = self.table_center_z + self.table_size_z * 0.5
                gt.pose.position.z = float(table_top_z + self.cube_size * 0.5)
                self.cube_pose_pub.publish(gt)

        if goal_bbox is not None:
            gx, gy, gw, gh = goal_bbox
            cv2.rectangle(cv_image, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 3)

            glabel = f"Goal {goal_score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(glabel, font, font_scale, thickness)
            y_top = max(gy - th - baseline - 6, 0)
            y_text = max(gy - baseline - 3, th + baseline + 3)
            cv2.rectangle(cv_image, (gx, y_top), (gx + tw + 6, gy), (0, 255, 0), -1)
            cv2.putText(cv_image, glabel, (gx + 3, y_text), font, font_scale, (255, 255, 255), thickness)

            u = gx + gw / 2.0
            v = gy + gh / 2.0
            container_top_z = self.container_center_z + self.container_size_z * 0.5
            hit = None if (self.use_gazebo_ground_truth and self.ground_truth_only) else self._intersect_plane_z(u, v, container_top_z, w_eff, h_eff)
            if hit is not None:
                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self.world_frame
                pose.pose.position.x = float(hit[0])
                pose.pose.position.y = float(hit[1])
                pose.pose.position.z = float(container_top_z)
                pose.pose.orientation.w = 1.0
                self.goal_pose_pub.publish(pose)
            elif self.use_gazebo_ground_truth:
                gt = self._get_link_pose_world(self.goal_link_name) or self._get_model_pose_world(self.goal_model_name)
                if gt is not None:
                    gt.pose.position.z = float(container_top_z)
                    self.goal_pose_pub.publish(gt)
        elif self.use_gazebo_ground_truth:
            gt = self._get_link_pose_world(self.goal_link_name) or self._get_model_pose_world(self.goal_model_name)
            if gt is not None:
                container_top_z = self.container_center_z + self.container_size_z * 0.5
                gt.pose.position.z = float(container_top_z)
                self.goal_pose_pub.publish(gt)

        try:
            out = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            out.header = msg.header
            self.image_pub.publish(out)
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge publish error: {e}")


def main() -> None:
    try:
        NeuralNetworkCamera()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

