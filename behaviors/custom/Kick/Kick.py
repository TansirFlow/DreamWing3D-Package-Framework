from agent.Base_Agent import Base_Agent
from behaviors.custom.Kick.Env import Env
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import numpy as np
import pickle
from behaviors.custom.Step.Step_Generator import Step_Generator
from encrypt.Encrypt import Encrypt
import os

class Kick():

    def __init__(self, base_agent: Base_Agent) -> None:
        self.phase = None
        self.reset_time = None
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "RL dribble"
        self.auto_head = True
        self.env = Env(base_agent)
        self.kick_flag = 0
        r_type = self.world.robot.type
        self.bias_dir = [0.09, 0.1, 0.14, 0.08, 0.05][self.world.robot.type]
        self.ball_x_center = 0.20
        self.ball_y_center = -0.04
        with open(M.get_active_directory([
            "/pkl/r0/Kick_R0.pkl",
            "/pkl/r1/Kick_R1.pkl",
            "/pkl/r2/Kick_R2.pkl",
            "/pkl/r3/Kick_R3.pkl",
            "/pkl/r4/Kick_R4.pkl"
        ][self.world.robot.type]), 'rb') as f:
            if os.path.exists("directReadFlag.txt"):
                self.model = pickle.load(f)
            else:
                cipher_text=f.read()
                plain_text = Encrypt.decrypt(cipher_text)
                self.model = pickle.loads(plain_text)
    
    def execute(self, reset, direction, abort=False):
        '''
        Parameters
        ----------
        orientation : float
            absolute or relative orientation of torso (relative to imu_torso_orientation), in degrees
            set to None to dribble towards the opponent's goal (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        speed : float
            speed from 0 to 1 (scale is not linear)
        stop : bool
            return True immediately if walking, wind down if dribbling, and return True when possible
        '''

        w = self.world
        r = self.world.robot
        b = w.ball_rel_torso_cart_pos
        t = w.time_local_ms
        gait: Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator

        if reset:
            self.kick_flag = 0
            self.phase = 0
            self.reset_time = t


        if self.phase == 0:
            biased_dir = M.normalize_deg(direction + self.bias_dir)  # add bias to rectify direction
            ang_diff = abs(
                M.normalize_deg(biased_dir - r.loc_torso_orientation))  # the reset was learned with loc, not IMU

            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)

            if (w.ball_last_seen > t - w.VISUALSTEP_MS and ang_diff < 5 and
                    t - w.ball_abs_pos_last_update < 100 and  # ball absolute location is recent
                    dist_to_final_target < 0.025 and  # if absolute ball position is updated
                    not gait.state_is_left_active and gait.state_current_ts == 2):  # to avoid kicking immediately without preparation & stability
                self.phase = 1
                self.env.kick_ori = direction
                obs = self.env.observe(True)
                action = run_mlp(obs, self.model)
                self.behavior.state_sub_behavior_name = None
                self.env.execute(action)
            else:
                dist = max(0.07, dist_to_final_target)
                reset_walk = reset and self.behavior.previous_behavior != "Walk"  # reset walk if it wasn't the previous behavior
                self.behavior.state_sub_behavior_name = "Walk"
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True,
                                                   dist)  # target, is_target_abs, ori, is_ori_abs, distance

                self.phase = 0
                return abort  # abort only if self.phase == 0
        else:
            if (self.phase > 20):
                return abort
            self.env.kick_ori = direction
            obs = self.env.observe(False)
            action = run_mlp(obs, self.model)
            self.env.execute(action)
            self.phase += 1

    def is_ready(self):
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True