#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

from functools import cache, cached_property
from typing import Literal, Optional, cast

import numpy as np
import pinocchio as pin  # type: ignore
import upkie_description  # type: ignore


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

        self.base_name = "base"

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
        self.joint_names = [name for name in self.model.names]

        self._log_dict = {}

    @cached_property
    def base_idx_q(self) -> list[int]:
        """Get the index of the base joints in the configuration space.

        Returns:
            The index of the base joints.
        """
        return [i for i in range(7)]

    @cached_property
    def base_idx_v(self) -> list[int]:
        """Get the index of the base joints in the velocity space.

        Returns:
            The index of the base joints.
        """
        return [i for i in range(6)]

    @cache
    def get_leg_idx(
        self,
        leg: Literal["left", "right", "both"],
        base_offset: Literal["no_base", "tangent", "config"] = "tangent",
    ) -> list[int]:
        """Get the index of the leg.

        Args:
            leg: The leg name, either "left", "right" or "both".
            base_offset: The index offset of the base joint. 'no_base' for no \
                offset, 'tangent' if indices are desired for the tangent \
                space and 'config' if indices are desired for the \
                configuration space.

        Returns:
            List of joint indices of the leg.
        """
        joint_idx = []

        offset = {"no_base": 0, "tangent": 6, "config": 7}[base_offset]

        # remove the 2 first joints (root and floating base)
        offset = offset - 2

        # print joint idx and names
        for joint_name in self.joint_names:
            if leg in joint_name or leg == "both":
                joint_id = self.model.getJointId(joint_name)
                joint_idx.append(joint_id + offset)

        return joint_idx

    def compute(
        self, q: np.ndarray, v: Optional[np.ndarray], a: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the inverse dynamics of the robot.

        Args:
            q: The joint configuration.
            v: The joint velocity.
            a: The joint acceleration.

        Returns:
            The joint torques, both with and without contact forces.
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

        # Compute gravity
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        if is_null(a):
            tau_l_M = np.zeros(3)
        else:
            # mypy hint: a is *always* np.ndarray if we get here
            a = cast(np.ndarray, a)

            # Get the reduced mass matrix M_l_lb (i.e., the mass matrix
            # of the robot that maps the base and the chosen leg onto
            # the chosen leg)
            leg_idx = self.get_leg_idx("left", "tangent")

            # M_l_lb s.t. M_l_lb * [a_b; a_l] = M_l * a_l + M_lb * a_b
            M_l_lb = self.data.M[leg_idx, :][:, self.base_idx_v + leg_idx]

            # Forces on the leg due to joint accelerations
            tau_l_M = M_l_lb @ a[self.base_idx_v + leg_idx]

        # Compute the non-linear effects (gravity, Coriolis, centrifugal)
        tau_l_nle = self.data.nle[leg_idx]

        # Compute the joint torques
        tau_no_contact = tau_l_M + tau_l_nle + g[leg_idx]

        # Compute the contact forces and project them onto the joints
        tau_contact = tau_no_contact + self.contact_on_joints(q, v, a)

        return tau_no_contact, tau_contact

    def contact_on_joints(
        self, q: np.ndarray, v: Optional[np.ndarray], a: Optional[np.ndarray]
    ) -> np.ndarray:
        r"""Compute the contact forces and project them onto the joints.

        This method computes $J^T_{fl} F_{fl}$, where $J_{fl}$ is the \
        contact Jacobian and $F_{fl}$ is the contact force, as given in \
        Eq. (5, 6) of the Hwangbo et al. paper.

        Only point contacts are considered, so the contact forces are \
        assumed to be linear forces (i.e., no torques).

        Args:
            q: The joint configurations.
            v: The joint velocities (optional).
            a: The joint accelerations (optional).

        Returns:
            The contact forces on the joints.
        """
        # Compute the contact Jacobians
        # Assume only linear forces, no torques (i.e., 3D point contact)
        # (3, nv)
        ref_frame = pin.LOCAL_WORLD_ALIGNED
        foot_frame_id = self.model.getFrameId(self.left_foot_frame)
        leg_idx = self.get_leg_idx("left", "tangent")

        J = pin.computeFrameJacobian(
            self.model, self.data, q, foot_frame_id, ref_frame
        )

        J_f = J[:3, self.base_idx_v + leg_idx]  # (3, 9)
        J_fl = J_f[:3, leg_idx]  # (3, 3)


        if not is_null(v):
            # Compute the contact Jacobian time variation
            # (3, nv)
            Jdot = pin.frameJacobianTimeVariation(
                self.model, self.data, q, v, foot_frame_id, ref_frame
            ) # (6, nv)

            Jdot_f = Jdot[:3, self.base_idx_v + leg_idx]  # (3, 6)

        joint_forces = np.zeros(3)

        if not is_null(v):
            v = cast(np.ndarray, v)  # mypy hint: v is *always* np.ndarray here
            joint_forces += Jdot_f.dot(v[self.base_idx_v + leg_idx])

        if not is_null(a):
            a = cast(np.ndarray, a)
            joint_forces += J_f.dot(a[self.base_idx_v + leg_idx])

        # Project the contact forces onto the joints
        J_fl = J_f[:3, leg_idx]  # (3, 3)
        J_fl_inv = np.linalg.pinv(J_fl)  # (3, 3)

        M_l = self.data.M[leg_idx, leg_idx]  # (3, 3)

        tau_contact = M_l @ J_fl_inv @ joint_forces

        return tau_contact

    def cycle(self, observation: dict, dt: float) -> dict:
        """Compute the inverse dynamics of the given leg."""
        # Populate the q, v, a arrays from the observation
        joint_idx = self.get_leg_idx("both", "tangent")
        for joint_id, joint_name in zip(joint_idx, self.joint_names):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                q = observation["servo"][joint_name]["position"]
                v = observation["servo"][joint_name]["velocity"]
                tau = observation["servo"][joint_name]["torque"]

                # Finite differences to compute the acceleration
                a = v - self._v[joint_id] if dt > 0 else 0.0
                a = a / dt

                # Update the values
                self._q[joint_id] = q
                self._v[joint_id] = v
                self._a[joint_id] = a

                # Update the measured torques
                self.tau_measured[joint_id] = tau

        # Fill in the base joint values
        self._q[:3] = 0.0  # We never know the base position
        self._q[3:7] = observation["imu"]["orientation"]

        # Fill in the base velocity values
        self._v[:3] = 0.0  # We never know the base velocity
        # (we could integrate but it would drift)
        self._v[3:6] = observation["imu"]["angular_velocity"]

        # Fill in the base acceleration values
        self._a[:3] = observation["imu"]["linear_acceleration"]
        self._a[3:6] = 0.0

        # Compute the inverse dynamics
        tau_no_contact, tau_contact = self.compute(self._q, self._v, self._a)

        leg_idx = self.get_leg_idx("left", "tangent")

        contact_error = tau_contact - self.tau_measured[leg_idx]
        no_contact_error = tau_no_contact - self.tau_measured[leg_idx]

        self._log_dict = {
            "tau_no_contact": tau_no_contact,
            "tau_contact": tau_contact,
            "contact_error": contact_error,
            "no_contact_error": no_contact_error
        }

        return self._log_dict

    def log(self) -> dict:
        """Return the log dictionary."""
        return self._log_dict

