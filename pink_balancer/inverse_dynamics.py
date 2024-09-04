from functools import cache
from typing import Literal

import numpy as np
import pinocchio as pin
import upkie_description


class InverseDynamics:
    """Compute the inverse dynamics of the robot."""
    def __init__(self):
        """Initialize the robot model."""
        self.robot = upkie_description.load_in_pinocchio(
            root_joint=pin.JointModelFreeFlyer()
        )

    @cache
    def get_leg_idx(
        self, leg: Literal["left", "right"], base_offset: Literal[0, 6, 7]
    ) -> tuple[int]:
        """Get the index of the leg.

        Args:
            leg: The leg name, either "left" or "right".
            base_offset: The index offset of the base joint. 0 for no offset, \
            6 if indices are desired for the tangent space and 7 if indices \
            are desired for the configuration space.

        Returns:
            The index of the leg joints
        """
        joint_idx = []

        # remove the 2 first joints (root and floating base)
        offset = base_offset - 2

        # print joint idx and names
        for joint_name in self.model.names:
            if leg in joint_name:
                joint_id = self.model.getJointId(joint_name)
                joint_idx.append(joint_id + offset)

        return tuple(joint_idx)
