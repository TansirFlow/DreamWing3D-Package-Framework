
from agent.Base_Agent import Base_Agent

import numpy as np
from math_ops.Math_Ops import Math_Ops as U
from world.World import World


class Env:

    def __init__(self,base_agent: Base_Agent, world=None):
        self.base_agent = base_agent
        self.world = base_agent.world
        self.obs = np.zeros(63, np.float32)
        self.DEFAULT_ARMS = np.array([-90,-90,8,8,90,90,70,70], np.float32)
        self.kick_ori = None
        self.kick_dist = None

    def observe(self, init=(False,)):
        w = self.world
        r = self.world.robot
        if init:
            self.step_counter = 0
            self.act = np.zeros(16, np.float32)
        self.obs[0] = self.step_counter / 20
        self.obs[1] = r.loc_head_z * 3
        self.obs[2] = r.loc_head_z_vel / 2
        self.obs[3] = r.imu_torso_roll / 15
        self.obs[4] = r.imu_torso_pitch / 15
        self.obs[5:8] = r.gyro / 100
        self.obs[8:11] = r.acc / 10
        self.obs[11:17] = r.frp.get('lf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)
        self.obs[17:23] = r.frp.get('rf', np.zeros(6)) * (10, 10, 10, 0.01, 0.01, 0.01)
        self.obs[23:39] = r.joints_position[2:18] / 100
        self.obs[39:55] = r.joints_speed[2:18] / 6.1395
        ball_rel_hip_center = self.base_agent.inv_kinematics.torso_to_hip_transform(w.ball_rel_torso_cart_pos)
        if init:
            self.obs[55:58] = (0, 0, 0)
        elif w.ball_is_visible:
            self.obs[55:58] = (ball_rel_hip_center - self.obs[58:61]) * 10
        self.obs[58:61] = ball_rel_hip_center
        self.obs[61] = np.linalg.norm(ball_rel_hip_center) * 2
        self.obs[62] = U.normalize_deg(self.kick_ori - r.imu_torso_orientation) / 30
        return self.obs

    def execute(self, action):
        w = self.world
        r = self.world.robot
        r.joints_target_speed[2:18] = action

        r.set_joints_target_position_direct([0, 1], np.array([0, -44], float), False)
        self.step_counter += 1
        return self.step_counter >= 22
