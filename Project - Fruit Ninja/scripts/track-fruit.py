#!/usr/bin/env python

import yaml
import rospy
import rospkg
import tf2_ros
import tf_conversions
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker

from std_msgs.msg import Float32

import cv2
import numpy as np
from ultralytics import YOLO

class trackFruit:
    def __init__(self):
        rospy.init_node('track_fruit', anonymous=True)

        # Load config
        self.loadConfig()
        
        # Load model
        if self.verbose: rospy.loginfo("Loading model...")
        self.model = YOLO(self.model_path)

        # Get camera info for 3D conversion
        if self.verbose: rospy.loginfo("Waiting for camera_info...")
        camera_info = rospy.wait_for_message('rgb/camera_info', CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape((3, 3))

        # Initialize publishers and subscribers
        image_sub = message_filters.Subscriber('rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
        self.landing_pos_pub = rospy.Publisher('landing_pos', Point)

        # NOTE: Temporary publishers
        self.x_pub = rospy.Publisher('x_pos', Float32, queue_size=10)
        self.y_pub = rospy.Publisher('y_pos', Float32, queue_size=10)
        self.z_pub = rospy.Publisher('z_pos', Float32, queue_size=10)

        if self.visualize:
            self.yolo_pub = rospy.Publisher('yolo_detections',Image,queue_size=10)
            self.ball_pub = rospy.Publisher("ball_marker", Marker, queue_size=10)
            self.landing_pub = rospy.Publisher("landing_marker", Marker, queue_size=10)
            
            # Define ball marker
            self.ball_marker = Marker()
            self.ball_marker.header.frame_id = "panda_link0"
            self.ball_marker.type = 2
            self.ball_marker.id = 0
            self.ball_marker.color.r = 1.0
            self.ball_marker.color.g = 0.65
            self.ball_marker.color.b = 0.0
            self.ball_marker.color.a = 1.0
            self.ball_marker.scale.x = 0.05
            self.ball_marker.scale.y = 0.05
            self.ball_marker.scale.z = 0.05
            self.ball_marker.pose.orientation.w = 1

            # Define landing marker
            self.landing_marker = Marker()
            self.landing_marker.header.frame_id = "panda_link0"
            self.landing_marker.type = 3
            self.landing_marker.id = 0
            self.landing_marker.color.r = 1.0
            self.landing_marker.color.g = 0.0
            self.landing_marker.color.b = 0.0
            self.landing_marker.color.a = 1.0
            self.landing_marker.scale.x = 0.05
            self.landing_marker.scale.y = 0.05
            self.landing_marker.scale.z = 0.01
            self.landing_marker.pose.orientation.w = 1

        # Create time synchronizer for image and depth
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=5, slop=0.2)
        ts.registerCallback(self.callback)

        self.cv_bridge = CvBridge()
        self.pos_hist = [(np.zeros(3), 0) for i in range(self.position_history_length)]
        self.pos_hist_filt = [(np.zeros(3), 0) for i in range(self.position_history_length)]
        self.landing_pos_hist = [Point(x=np.inf, y=np.inf, z=np.inf) for i in range(self.position_history_length + 1)]
        self.landing_pos = Point(x=np.inf, y=np.inf, z=np.inf)

        # Get camera transform
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        trans = tfBuffer.lookup_transform("panda_link0", "camera_base", rospy.Time(), rospy.Duration.from_sec(0.5)).transform
        pose = Pose(position=Point(x=trans.translation.x, y=trans.translation.y, z=trans.translation.z), orientation=trans.rotation)
        self.cam_to_world = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        if self.verbose: rospy.loginfo("Done!")

    def loadConfig(self):
        rospack = rospkg.RosPack()
        config_path = rospack.get_path('fruit-ninja-vision') + '/config/default.yaml'

        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.visualize = config["visualize"]
        self.verbose = config["verbose"]
        self.model_path = config["model_path"]
        self.sub_box_scale = config["sub_box_scale"]
        self.position_history_length = config["position_history_length"]
        self.plane_height = config["plane_height"]

    def callback(self, image_msg, depth_msg):
        # Convert images to arrays
        img = self.cv_bridge.imgmsg_to_cv2(image_msg)[:, :, :3]
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        _, r, _ = img.shape

        # Run inference
        result = self.model.predict(source=img, conf=0.1, verbose=self.verbose)[0]

        # Determine bounding box
        out_img = cv2.rotate(img.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        if len(result.boxes) > 0:
            box = result.boxes[0].xyxy.squeeze()
            out_img = cv2.rectangle(out_img, (int(box[3]), int(r - box[2])), (int(box[1]), int(r - box[0])), (255, 0, 0))

            # Crop to center of box
            lx = box[3] - box[1]
            ly = box[2] - box[0]
            x_offset = (lx - (lx / self.sub_box_scale) ) / 2
            y_offset = (ly - (ly / self.sub_box_scale) ) / 2

            y1 = int(box[1] + x_offset)
            x1 = int(box[0] + y_offset)
            y2 = int(box[3] - x_offset)
            x2 = int(box[2] - y_offset)

            # Get median of depth
            depth_sample = depth[y1:y2, x1:x2]
            med_depth = np.median(depth_sample[depth_sample > 0])

            if np.isnan(med_depth): return

            # Determine center of box in pixel coords
            px_x = box[0] + lx / 2
            px_y = box[1] + ly / 2

            # Determine ball location in 3D
            x = med_depth
            y = -(px_x - self.intrinsic[0, 2]) / self.intrinsic[0, 0] * med_depth
            z = -(px_y - self.intrinsic[1, 2]) / self.intrinsic[1, 1] * med_depth

            # Convert location to world frame
            x, y, z, _ = np.matmul(self.cam_to_world, np.array([x / 1000, y.item() / 1000 - 0.032, z.item() / 1000, 1]))

            # Update position history
            self.pos_hist.pop(0)
            self.pos_hist.append((np.array([x, y, z]), rospy.get_time()))

            # Filter position
            x_filt, y_filt, z_filt = np.median([pos for pos, _ in self.pos_hist], axis=0)

            # Update filtered position history
            self.pos_hist_filt.pop(0)
            self.pos_hist_filt.append((np.array([x_filt, y_filt, z_filt]), rospy.get_time()))

            if self.visualize:
                self.ball_marker.header.stamp = rospy.Time.now()
                self.ball_marker.pose.position.x = x_filt
                self.ball_marker.pose.position.y = y_filt
                self.ball_marker.pose.position.z = z_filt
                self.ball_pub.publish(self.ball_marker)

            # Calculate average velocity over last POSITION_HISTORY_LENGTH steps
            vels = np.zeros((self.position_history_length - 1, 3))
            for i in range(self.position_history_length - 1):
                new_pos, new_time = self.pos_hist_filt[i + 1]
                old_pos, old_time = self.pos_hist_filt[i]

                vels[i] = (new_pos - old_pos) / (new_time - old_time)

            vel = np.mean(vels, axis=0)

            # Get the landing time
            t = (-vel[2] - np.sqrt(vel[2] ** 2 + 4 * 4.9 * (z - self.plane_height))) / -9.8

            # Get the landing position
            if (vel[2] ** 2 + 4 * 4.9 * (z - self.plane_height)) > 0 and vel[2] <= 0:
                self.landing_pos = Point(x=x_filt + vel[0] * t, y=y_filt + vel[1] * t, z=self.plane_height)

                # Update landing position history
                self.landing_pos_hist.pop(0)
                self.landing_pos_hist.append(self.landing_pos)

            if self.visualize:
                self.landing_marker.header.stamp = rospy.Time.now()
                self.landing_marker.pose.position = self.landing_pos
                self.landing_pub.publish(self.landing_marker)

                self.x_pub.publish(Float32(self.landing_pos.x))
                self.y_pub.publish(Float32(self.landing_pos.y))

            # Publish the landing position
            if self.landing_pos.x != np.inf:
                self.landing_pos_pub.publish(self.landing_pos)
            
        if self.visualize:
            self.yolo_pub.publish(self.cv_bridge.cv2_to_imgmsg(out_img))

if __name__ == "__main__":
    track_fruit = trackFruit()
    rospy.spin()