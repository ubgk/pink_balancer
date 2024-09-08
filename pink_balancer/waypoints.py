import numpy as np
import pinocchio as pin

# Waypoints to follow
WAYPOINTS: list[pin.SE3] = [
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.00, 0.00, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([1.00, 0.00, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([2.00, 0.00, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([2.50, 0.00, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([2.50, 0.80, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([3.5, 0.80, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([3.5, -2.5, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([2.50, -2.5, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([1.25, -2.5, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.50, -2.5, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.00, -2.5, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.00, -1.2, 0.00])),
    pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.00, -0.5, 0.00])),


]

def get_commands(observation: dict, waypoint_idx: int) -> tuple:
    """Get 2D twist to apply to the robot."""
    # Get the current position and orientation of the robot
    position = observation["sim"]["base"]["position"]
    (w, x, y, z) = observation["sim"]["base"]["orientation"]
    quat = pin.Quaternion(np.array([x, y, z, w]))
    pose = pin.SE3(quat, np.array(position))  # current pose in world frame

    print()
    print()
    print(f"{position=}")

    forward_vec = quat.matrix()[:, 0]
    robot_yaw = np.arctan2(forward_vec[1], forward_vec[0])

    # Get the desired position and orientation of the robot
    waypoint = WAYPOINTS[waypoint_idx]

    # Difference between the current and desired position and orientation
    waypoint_in_base = pose.inverse() * waypoint

    # Ignore the orientation difference
    waypoint_in_base = pin.SE3(pin.Quaternion(1, 0, 0, 0),
                               waypoint_in_base.translation)

    twist = pin.log(waypoint_in_base).vector
    twist = twist[:2]

    print(f"{twist=}")

    # Convert the twist to linear and angular velocities
    r = np.linalg.norm(twist)
    theta = np.arctan2(twist[1], twist[0])
    # theta = theta - np.pi / 2

    print(f"{r=}, {theta=}")

    if np.abs(r) < 0.1:
        print(f"Reached waypoint {waypoint_idx}")
        waypoint_idx = (waypoint_idx + 1) % len(WAYPOINTS)


    # Prepare the commands to send to the robot
    linear_velocity = 1.0
    angular_velocity = -theta * 5.0

    linear_velocity = np.clip(linear_velocity, -1.0, 1.0)
    angular_velocity = np.clip(angular_velocity, -1.0, 1.0)

    right_trigger = 1.0

    # We want to keep the robot orientation close to the desired orientation
    if np.abs(theta) > np.pi / 4:
        linear_velocity = 0.0

    cmd_dict = {
        "left_axis": [0.0, -linear_velocity],
        "right_axis": [angular_velocity, 0.0],
        "right_trigger": right_trigger,
        "pad_axis": [0.0, 1.0],
    }

    return cmd_dict, waypoint_idx


