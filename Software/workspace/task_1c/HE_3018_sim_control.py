#!/usr/bin/env python3
"""
Boilerplate controller for the HE bot.

Fetches a shape from the get_shape service and builds a list of waypoints
to trace it. Fill in the control loop to drive the robot through them.


"""

import argparse
import math
from turtle import speed

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape

# Wheel <-> body-velocity mapping, columns are [left, right, back] wheel
# speed (rad/s); rows are body frame [vx, vy, wz] per unit wheel speed.
WHEEL_RADIUS = 0.0255
CHASSIS_RADIUS = 0.06412
WHEEL_ANGLES = np.radians([30.0, 150.0, 270.0])

_BODY_TO_WHEEL = np.array([
    [np.cos(WHEEL_ANGLES[0]), np.sin(WHEEL_ANGLES[0]), CHASSIS_RADIUS],
    [np.cos(WHEEL_ANGLES[1]), np.sin(WHEEL_ANGLES[1]), CHASSIS_RADIUS],
    [np.cos(WHEEL_ANGLES[2]), np.sin(WHEEL_ANGLES[2]), CHASSIS_RADIUS],
]) / WHEEL_RADIUS

_WHEEL_TO_BODY = np.linalg.inv(_BODY_TO_WHEEL)
_CTRL_LIMIT = 10.0      # rad/s, matches lekiwi.xml actuator ctrlrange

WAYPOINT_TOLERANCE = 0.03   # metres
CIRCLE_SEGMENTS     = 36
POSITION_KP         = 2.0
YAW_HOLD_KP         = 1.5
CONTROL_PERIOD      = 0.05  

def body_to_wheels(vx, vy, wz):
    """Body-frame (vx, vy, wz) -> wheel angular velocities [left, right, back]."""

    body_velocity = np.array([vx, vy, wz], dtype=float)

    wheels = _BODY_TO_WHEEL @ body_velocity

    wheels = np.clip(wheels, -_CTRL_LIMIT, _CTRL_LIMIT)

    return wheels.tolist()


def yaw_from_quat(w, x, y, z):
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    """Vertices of a regular polygon centred on (cx, cy), closed back to the
    first vertex so the last waypoint returns the robot to where it started
    drawing."""
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    """World-frame waypoints for `shape_name`, as returned by the get_shape
    service: data[0:2] is the shape's centre (x, y); the remaining entries
    are its size parameters (see shape_service.cpp's shape_map)."""
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]

    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)

    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])

    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])

    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"unknown shape '{shape_name}'")


class ShapeController(Node):
#     def __init__(self, speed):
#         super().__init__("shape_controller")
#         self.speed = speed

#         self.pose = None        # (x, y, yaw), latest ground truth
#         self.start_pose = None  # (x, y, yaw), recorded on first odom message
#         self.wp_index = 0
#         self.done = False
# #Add the publsiher and subscriber scripts

    def __init__(self, speed):
        super().__init__("shape_controller")

        self.speed = speed

        self.pose = None
        self.start_pose = None
        self.wp_index = 0
        self.done = False

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/wheel_commands",
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_cb,
            10
        )

        shape_name, waypoints = self._request_shape()

        self.shape_name = shape_name
        self.waypoints = waypoints

        self.get_logger().info(
            f"Assigned shape: {shape_name}"
        )

        self.get_logger().info(
            f"Waypoints: {len(waypoints)}"
        )

        self.timer = self.create_timer(
            CONTROL_PERIOD,
            self._control_step
        )
        
    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"get_shape service call failed: {response and response.message}"
            )

        return response.shape_name, build_waypoints(response.shape_name, list(response.data))

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        yaw = yaw_from_quat(q.w, q.x, q.y, q.z)

        self.pose = (p.x, p.y, yaw)

        if self.start_pose is None:
            self.start_pose = self.pose

    def _publish(self, wheels):
        self.cmd_pub.publish(Float64MultiArray(data=wheels))

    def _control_step(self):
        if self.done or self.pose is None:
            return

        if self.wp_index >= len(self.waypoints):
            self._publish([0.0, 0.0, 0.0])
            self.done = True
            return

        # Current pose
        x, y, yaw = self.pose

        # Current waypoint
        target_x, target_y = self.waypoints[self.wp_index]

        # Position error in world frame
        error_x = target_x - x
        error_y = target_y - y

        distance = math.hypot(error_x, error_y)

        # Check whether waypoint is reached
        if distance <= WAYPOINT_TOLERANCE:
            self.wp_index += 1

            if self.wp_index >= len(self.waypoints):
                self._publish([0.0, 0.0, 0.0])
                self.done = True
                self.get_logger().info("Shape completed!")
                return

            target_x, target_y = self.waypoints[self.wp_index]

            error_x = target_x - x
            error_y = target_y - y

        # Convert world-frame position error into robot/body frame
        body_error_x = (
            math.cos(yaw) * error_x
            + math.sin(yaw) * error_y
        )

        body_error_y = (
            -math.sin(yaw) * error_x
            + math.cos(yaw) * error_y
        )

        # Proportional position controller
        vx = POSITION_KP * body_error_x
        vy = POSITION_KP * body_error_y

        # Limit translational speed
        speed_mag = math.hypot(vx, vy)

        if speed_mag > self.speed:
            scale = self.speed / speed_mag
            vx *= scale
            vy *= scale

        # Hold current heading
        wz = 0.0

        wheels = body_to_wheels(vx, vy, wz)

        self._publish(wheels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.25,
                         help="max approach speed, m/s")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ShapeController(args.speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
