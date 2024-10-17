#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

from functools import cache, cached_property
from typing import Literal, Optional, cast

import numpy as np
import pinocchio as pin  # type: ignore
import upkie_description  # type: ignore
from upkie import model  # type: ignore

upkie_model = model.Model(upkie_description.URDF_PATH)
R_BASE_TO_IMU: np.ndarray = upkie_model.rotation_base_to_imu
R_IMU_TO_BASE: np.ndarray = R_BASE_TO_IMU.T
R_ARS_TO_WORLD: np.ndarray = upkie_model.rotation_ars_to_world

# We also provide the above in quaternion form
Q_BASE_TO_IMU: pin.Quaternion = pin.Quaternion(R_BASE_TO_IMU)
Q_IMU_TO_BASE: pin.Quaternion = pin.Quaternion(R_IMU_TO_BASE)
Q_ARS_TO_WORLD: pin.Quaternion = pin.Quaternion(R_ARS_TO_WORLD)


def is_null(vec: Optional[np.ndarray]) -> bool:
    """Check if a vector is null or NoneType.

    Args:
        vec: The vector to check.

    Returns:
        True if the vector is null or the argument is None,\
            False otherwise.
    """
    if vec is None:
        return True

    return np.allclose(vec, 0)


class InverseDynamics:
    """Compute the inverse dynamics of the robot."""

    def __init__(self):
        """Initialize the robot model."""
        self.robot = upkie_description.load_in_pinocchio(
            root_joint=pin.JointModelFreeFlyer()
        )

        self.model = self.robot.model
        self.data = self.robot.data

        self.root_name = "root_joint"

        # Contact frames, we don't care about wheels now and assume the \
        # robot is standing on flat ground.
        self.left_foot_frame = "left_anchor"
        self.right_foot_frame = "right_anchor"

        # Initialize the state vectors
        self._q = np.zeros(self.model.nq)
        self._v = np.zeros(self.model.nv)
        self._a = np.zeros(self.model.nv)

        # Initialize the measured torques
        self.tau_measured = np.zeros(self.model.nv)

        # Initialize joint names
        self.joint_names = [
            name for name in self.model.names if name != "universe"
        ]

        self._log_dict = {}

    @cached_property
    def base_indices_q(self) -> list[int]:
        """Get the indices of the base joints in the configuration space.

        Returns:
            The indices of the base joints.
        """
        joint = self.model.joints[self.model.getJointId(self.root_name)]

        return [i for i in range(joint.idx_q, joint.nq)]

    @cached_property
    def base_indices_v(self) -> list[int]:
        """Get the indices of the base joints in the velocity space.

        Returns:
            The indices of the base joints.
        """
        joint = self.model.joints[self.model.getJointId(self.root_name)]

        return [i for i in range(joint.idx_v, joint.nv)]

    @cache
    def get_leg_indices(
        self,
        leg: Literal["left", "right", "both"],
        space: Literal["config", "tangent"] = "tangent",
    ) -> list[int]:
        """Get the indices of the leg.

        Args:
            leg: The leg name, either "left", "right" or "both".
            space: The space to get the indices from, either "config" or\
                "tangent", which corresponds to the configuration and\
                tangent spaces, respectively. Defaults to "tangent".

        Returns:
            List of joint indices of the leg.
        """
        joint_indices = []

        for joint_name in self.joint_names:
            if leg in joint_name or (
                leg == "both" and joint_name != "root_joint"
            ):
                joint_id = self.model.getJointId(joint_name)
                joint = self.model.joints[joint_id]

                if space == "config":
                    joint_indices.append(joint.idx_q)
                elif space == "tangent":
                    joint_indices.append(joint.idx_v)
                else:
                    raise ValueError(f"Invalid space: {space}! Must be\
                                      'config' or 'tangent'.")

        return joint_indices

    def compute_torques(
        self, q: np.ndarray, v: Optional[np.ndarray], a: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the inverse dynamics of the robot.

        Args:
            q: The joint configuration.
            v: The joint velocity.
            a: The joint acceleration.

        Returns:
            The estimated joint torques, both with and without contact forces.
        """
        assert q.shape == (
            self.model.nq,
        ), "Invalid joint configuration shape."

        if v is not None:
            assert v.shape == (self.model.nv,), "Invalid joint velocity shape."

        if a is not None:
            assert a.shape == (
                self.model.nv,
            ), "Invalid joint acceleration shape."

        # Update the robot state
        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.computeJointJacobians(self.model, self.data, q)

        # Compute the mass matrix
        pin.crba(self.model, self.data, q)

        # Compute gravity
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        leg_indices_v = self.get_leg_indices("left", "tangent")

        # Compute the joint torques
        tau_no_contact = (pin.rnea(self.model, self.data, q, v, a))[
            leg_indices_v
        ]

        # Compute the contact forces and project them onto the joints
        tau_contact = tau_no_contact + self.contact_torques(q, v, a)

        return tau_no_contact, tau_contact, g[leg_indices_v]

    def contact_torques(
        self, q: np.ndarray, v: Optional[np.ndarray], a: Optional[np.ndarray]
    ) -> np.ndarray:
        r"""Compute the contact forces and project them onto the joints.

        This method computes $J^T_{fl} F_{fl}$, where $J_{fl}$ is the \
        contact Jacobian and $F_{fl}$ is the contact force, applied at the
        foot (in our case, the leg anchor frame), as given in Eq. (5, 6)\
        of the Hwangbo et al. paper.

        Only point contacts are considered, so the contact forces are \
        assumed to be linear forces (i.e., no torques).

        Args:
            q: The joint configurations.
            v: The joint velocities (optional).
            a: The joint accelerations (optional).

        Returns:
            The joint torques due to contact forces.
        """
        # Compute the contact Jacobians
        # Assume only linear forces, no torques (i.e., 3D point contact)
        # (3, nv)
        ref_frame = pin.LOCAL_WORLD_ALIGNED
        foot_frame_id = self.model.getFrameId(self.left_foot_frame)
        leg_indices_v = self.get_leg_indices("left", "tangent")

        J = pin.computeFrameJacobian(
            self.model, self.data, q, foot_frame_id, ref_frame
        )

        J_f = J[:3, self.base_indices_v + leg_indices_v]  # (3, 9)
        J_fl = J[:3, leg_indices_v]  # (3, 3)
        J_fl_inv = np.linalg.pinv(J_fl)  # (3, 3)

        if not is_null(v):
            # Compute the contact Jacobian time variation
            # (3, nv)
            Jdot = pin.frameJacobianTimeVariation(
                self.model, self.data, q, v, foot_frame_id, ref_frame
            )  # (6, nv)

            Jdot_f = Jdot[:3, self.base_indices_v + leg_indices_v]  # (3, 6)

        # Joint torques due to contact forces
        joint_torques = np.zeros(3)

        if not is_null(v):
            v = cast(np.ndarray, v)  # mypy hint: v is *always* np.ndarray here
            joint_torques += Jdot_f.dot(v[self.base_indices_v + leg_indices_v])

        if not is_null(a):
            a = cast(np.ndarray, a)
            joint_torques += J_f.dot(a[self.base_indices_v + leg_indices_v])

        M_l = self.data.M[leg_indices_v, :][:, leg_indices_v]  # (3, 3)

        tau_contact = M_l @ J_fl_inv @ joint_torques

        return tau_contact

    def cycle(self, observation: dict, dt: float) -> dict:
        """Compute the inverse dynamics of the given leg."""
        # Populate the q, v, a arrays from the observation

        leg_indices_q = self.get_leg_indices("both", "config")
        leg_indices_v = self.get_leg_indices("both", "tangent")
        for joint_name, joint_idx_q, joint_idx_v in zip(
            self.joint_names[1:], leg_indices_q, leg_indices_v
        ):
            q = observation["servo"][joint_name]["position"]
            v = observation["servo"][joint_name]["velocity"]
            tau = observation["servo"][joint_name]["torque"]

            v = 0.95 * self._v[joint_idx_v] + 0.05 * v

            # Finite differences to compute the acceleration
            a = v - self._v[joint_idx_v] if dt > 0 else 0.0
            a = a / dt
            # a = 0

            # Update the values
            self._q[joint_idx_q] = q
            self._v[joint_idx_v] = v
            self._a[joint_idx_v] = a

            # Update the measured torques
            self.tau_measured[joint_idx_v] = tau

        # Fill in the base joint values
        self._q[:3] = 0.0  # We don't observe the base position, and torques\
        # are not affected by it

        # N.B.: Pinocchio and the IMU use different conventions for quaternions
        (w, x, y, z) = observation["imu"]["orientation"]
        Q_imu_to_ars = pin.Quaternion(np.array([x, y, z, w]))
        Q_imu_to_world = Q_ARS_TO_WORLD * Q_imu_to_ars
        self._q[3:7] = Q_imu_to_world.coeffs()  # Quaternion representation

        # Fill in the base velocity values
        self._v[:3] = 0.0  # We don't observe the base velocity
        # (we could integrate but it would drift)

        # We convert the angular velocity from the IMU frame to the base frame
        omega_imu = np.array(observation["imu"]["angular_velocity"])
        omega_base = R_IMU_TO_BASE @ omega_imu
        self._v[3:6] = omega_base

        # Fill in the base acceleration values
        lin_accel_imu = np.array(observation["imu"]["linear_acceleration"])
        # self._a[0:6] = 0.0  # We don't observe the angular acceleration
        # self._a[2] = - 9.81  # Gravity in the z direction

        self._a[:3] = R_IMU_TO_BASE @ lin_accel_imu

        print(self._a[:3])

        # Compute the inverse dynamics
        tau_no_contact, tau_contact, gravity = self.compute_torques(
            self._q, self._v, self._a
        )

        leg_indices_v = self.get_leg_indices("left", "tangent")

        contact_error = np.abs(tau_contact - self.tau_measured[leg_indices_v])
        no_contact_error = np.abs(
            tau_no_contact - self.tau_measured[leg_indices_v]
        )

        self._log_dict = {
            "a": self._a,
            "v": self._v,
            "q": self._q,
            "tau_measured": self.tau_measured,
            "tau_no_contact": tau_no_contact,
            "tau_contact": tau_contact,
            "contact_error": contact_error,
            "no_contact_error": no_contact_error,
            "gravity": gravity,
            "acceleration": self._a[leg_indices_v],
        }

        return self._log_dict

    def log(self) -> dict:
        """Return the log dictionary."""
        return self._log_dict


if __name__ == "__main__":
    import argparse

    import mpacklog  # type: ignore
    import msgpack  # type: ignore
    from tqdm import tqdm  # type: ignore

    parser = argparse.ArgumentParser(description="Full-body inverse dynamics "
                                     "using Pinocchio.")

    parser.add_argument(
        "input_path",
        type=str,
        help="The input file containing the observations.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="The output file to save the computed torques.",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    assert args.input_path.endswith(".mpack"), "Invalid input file format."

    if args.output_path is not None:
        assert args.output_path.endswith(
            ".mpack"
        ), "Invalid output file format."
    else:
        args.output_path = args.input_path.replace(
            ".mpack", "_whole_dynamics.mpack"
        )
        print(f"Output file: {args.output_path}")

    with open(args.input_path, "rb") as input_file:
        unpacker = msgpack.Unpacker(input_file, raw=False)
        logger = mpacklog.SyncLogger(args.output_path)

        inverse_dynamics = InverseDynamics()

        for i, obj in tqdm(enumerate(unpacker)):
            observation = obj["observation"]
            inverse_dynamics.cycle(observation, dt=0.001)
            log_dict = inverse_dynamics.log()

            observation["whole_dynamics"] = log_dict
            logger.put(obj)

            # Write the log every 1000 cycles (i.e., every second at 1 kHz)
            if i % 1000 == 0:
                logger.write()

        input_file.close()
        logger.write()  # Write the last logs
