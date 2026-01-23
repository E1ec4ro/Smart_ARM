#!/usr/bin/env python3
import cv2
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point


class NeuralNetworkCamera:
    def __init__(self) -> None:
        rospy.init_node("neural_network_camera", anonymous=True)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/overhead_camera/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/overhead_camera/image_detected", Image, queue_size=1)
        self.cube_position_pub = rospy.Publisher("/detected_cube_position", Point, queue_size=1)

        self.lower_red1 = np.array([0, 80, 60])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 80, 60])
        self.upper_red2 = np.array([180, 255, 255])

        rospy.loginfo("Red cube detector initialized (simple HSV)")

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

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge conversion error: {e}")
            return

        bbox, score = self.detect_red_cube(cv_image)

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

