from random import random

from agent.Base_Agent import Base_Agent as Agent
from behaviors.custom.Step.Step import Step
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np
from math_ops.Math_Ops import Math_Ops as U
from math_ops.Math_Ops import Math_Ops as M
from behaviors.custom.Step.Step_Generator import Step_Generator

'''
Objective:
Learn how to run forward using step primitive
----------
- class Basic_Run: implements an OpenAI custom gym
- class Train:  implements algorithms to train a new model or test an existing model
'''


class Kick(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        self.lock_flag = False
        self.sleep = 0
        self.reset_time = None
        self.behavior = None
        self.path_manager = None
        self.bias_dir = None
        self.robot_type = r_type
        self.kick_ori = 0
        self.terminal = False
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        self.step_counter = 0  # to limit episode size
        self.ball_pos = np.array([0, 0, 0])

        self.step_obj: Step = self.player.behavior.get_custom_behavior_object("Step")  # Step behavior object

        # State space
        obs_size = 63
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(obs_size, -np.inf, np.float32),
                                                high=np.full(obs_size, np.inf, np.float32), dtype=np.float32)

        # Action space
        MAX = np.finfo(np.float32).max
        self.no_of_actions = act_size = 16
        self.action_space = gym.spaces.Box(low=np.full(act_size, -MAX, np.float32),
                                           high=np.full(act_size, MAX, np.float32), dtype=np.float32)

        # Place ball far away to keep landmarks in FoV (head follows ball while using Step behavior)
        self.player.scom.unofficial_move_ball((14, 0, 0.042))

        self.ball_x_center = 0.20
        self.ball_y_center = -0.04
        self.player.scom.unofficial_set_play_mode("PlayOn")
        self.player.scom.unofficial_move_ball((0, 0, 0))

    def observe(self, init=False):
        w = self.player.world
        r = self.player.world.robot

        if init:
            self.step_counter = 0
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
        ball_rel_hip_center = self.player.inv_kinematics.torso_to_hip_transform(w.ball_rel_torso_cart_pos)
        if init:
            self.obs[55:58] = (0, 0, 0)
        elif w.ball_is_visible:
            self.obs[55:58] = (ball_rel_hip_center - self.obs[58:61]) * 10
        self.obs[58:61] = ball_rel_hip_center
        self.obs[61] = np.linalg.norm(ball_rel_hip_center) * 2
        self.obs[62] = U.normalize_deg(self.kick_ori - r.imu_torso_orientation) / 30
        '''
        Expected observations for walking parameters/state (example):
        Time step        R  0  1  2  0   1   2   3  4
        Progress         1  0 .5  1  0 .25  .5 .75  1
        Left leg active  T  F  F  F  T   T   T   T  T
        Parameters       A  A  A  B  B   B   B   B  C
        Example note: (A) has a step duration of 3ts, (B) has a step duration of 5ts
        '''
        return self.obs

    def sync(self):
        ''' Run a single simulation step '''
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def reset(self):
        # print("reset")
        '''
        Reset and stabilize the robot
        Note: for some behaviors it would be better to reduce stabilization or add noise
        '''
        self.lock_flag = False
        self.player.scom.unofficial_set_play_mode("PlayOn")
        Gen_ball_pos = [random() * 5 - 9, random() * 6 - 3, 0]
        Gen_player_pos = (random() * 3 + Gen_ball_pos[0], random() * 3 + Gen_ball_pos[1], 0.5)
        self.ball_pos = np.array(Gen_ball_pos)
        self.player.scom.unofficial_move_ball((Gen_ball_pos[0], Gen_ball_pos[1], Gen_ball_pos[2]))
        self.sleep = 0
        self.step_counter = 0
        self.behavior = self.player.behavior
        r = self.player.world.robot
        w = self.player.world
        t = w.time_local_ms
        self.path_manager = self.player.path_manager
        gait: Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator
        self.reset_time = t

        for _ in range(25):
            self.player.scom.unofficial_beam(Gen_player_pos, 0)  # beam player continuously (floating above ground)
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()

        # beam player to ground
        self.player.scom.unofficial_beam(Gen_player_pos, 0)
        r.joints_target_speed[
            0] = 0.01  # move head to trigger physics update (rcssserver3d bug when no joint is moving)
        self.sync()

        # stabilize on ground
        for _ in range(7):
            self.player.behavior.execute("Zero_Bent_Knees")
            self.sync()
        # walk to ball
        while True and w.time_local_ms - self.reset_time <= 50000:
            direction = 0
            if self.player.behavior.is_ready("Get_Up"):
                self.player.behavior.execute_to_completion("Get_Up")
            self.bias_dir = [0.09, 0.1, 0.14, 0.08, 0.05][r.type]
            biased_dir = M.normalize_deg(direction + self.bias_dir)  # add bias to rectify direction
            ang_diff = abs(
                M.normalize_deg(biased_dir - r.loc_torso_orientation))  # the reset was learned with loc, not IMU

            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)
            if (w.ball_last_seen > t - w.VISUALSTEP_MS and ang_diff < 5 and
                    t - w.ball_abs_pos_last_update < 100 and  # ball absolute location is recent
                    dist_to_final_target < 0.025 and  # if absolute ball position is updated
                    not gait.state_is_left_active and gait.state_current_ts == 2):  # to avoid kicking immediately without preparation & stability
                break
            else:
                dist = max(0.07, dist_to_final_target)
                reset_walk = self.behavior.previous_behavior != "Walk"  # reset walk if it wasn't the previous behavior
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True,
                                                   dist)  # target, is_target_abs, ori, is_ori_abs, distance

            self.sync()

        # memory variables
        self.lastx = r.cheat_abs_pos[0]
        self.act = np.zeros(self.no_of_actions, np.float32)

        return self.observe(True)

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.terminate()

    def step(self, action):
        r = self.player.world.robot
        b = self.player.world.ball_abs_pos
        w = self.player.world
        t = w.time_local_ms
        r.joints_target_speed[2:18] = action
        r.set_joints_target_position_direct([0, 1], np.array([0, -44], float), False)

        self.sync()  # run simulation step
        self.step_counter += 1
        self.lastx = r.cheat_abs_pos[0]

        # terminal state: the robot is falling or timeout
        if self.step_counter > 22:
            obs = self.observe()
            self.player.scom.unofficial_beam((-14.5, 0, 0.51), 0)  # beam player continuously (floating above ground)
            waiting_steps = 0
            high = 0
            while waiting_steps < 650:  # 假设额外等待5个步骤
                if w.ball_cheat_abs_pos[2] > high:
                    high = w.ball_cheat_abs_pos[2]
                self.sync()  # 继续执行仿真步骤
                waiting_steps += 1
            dis = np.linalg.norm(self.ball_pos - w.ball_cheat_abs_pos)
            reward = dis - abs(w.ball_cheat_abs_pos[1] - self.ball_pos[1]) + high*0.2
            # print(reward)
            self.terminal = True

        else:
            obs = self.observe()
            reward = 0
            self.terminal = False

        return obs, reward, self.terminal, {}


class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):

        # --------------------------------------- Learning parameters
        n_envs = min(14, os.cpu_count())
        n_steps_per_env = 128  # RolloutBuffer is of size (n_steps_per_env * n_envs)
        minibatch_size = 64  # should be a factor of (n_steps_per_env * n_envs)
        total_steps = 30000000
        learning_rate = 3e-4
        folder_name = f'Kick_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        # print("Model path:", model_path)

        # --------------------------------------- Run algorithm
        def init_env(i_env):
            def thunk():
                return Kick(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)

            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)  # include 1 extra server for testing

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:  # retrain
                model = PPO.load(args["model_file"], env=env, device="cuda", n_envs=n_envs, n_steps=n_steps_per_env,
                                 batch_size=minibatch_size, learning_rate=learning_rate)
            else:  # train new model
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size,
                            learning_rate=learning_rate, device="cuda")

            model_path = self.learn_model(model, total_steps, model_path, eval_env=eval_env,
                                          eval_freq=n_steps_per_env * 20, save_freq=n_steps_per_env * 20,
                                          backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)  # wait for child processes
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        # Uses different server and monitor ports
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Kick(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)

        try:
            self.export_model(args["model_file"], args["model_file"] + ".pkl",
                              False)  # Export to pkl to create custom behavior
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()