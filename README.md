# DreamWing3D-Package-Framework
> 这是一个用于将基于FCP底层的3D球队代码打包成二进制的框架

## 安装依赖
```bash
pip install nuitka
conda install libpython-static  # 如果你是在conda环境下使用请执行这一步
sudo apt install patchelf
```

## 原理
使用nuitka将python代码转换成cpp代码，然后打包，这种方式需要将pkl和xml等资源文件暴露，所以该框架对这些文件进行了加密处理

## 温馨提示
1. `directReadFlag.txt` 文件只存在与项目中，不存在于打包后的目录，如果该文件存在，程序不会对pkl和xml进行解密，如果不存在则会对pkl和xml进行解密，具体运行机制参考`behaviors/custom/Dribble/Dribble.py`中的读取pkl部分代码
2. 请把所有的 **pkl文件** 存放在项目根目录下`pkl`文件夹内，然后在动作类中指定pkl路径如`/pkl/r0/dribble_R0.pkl`
3. 您可以在`package/package.sh`脚本中修改队伍名称(teamname_name)和二进制名称(binary_name)
4. 生成的二进制文件及其依赖文件位于`package/target`
5. 根目录下的`libs`文件夹包含了一些共享库，请不要删除它，打包的时候会自动拷贝
6. 每次打包时脚本会自动更新加密秘钥，加密范围包括pkl文件和xml文件(world/commons/robots部分的xml不会进行加密)

## 感谢
FCPortugal团队 开源的底层，仓库地址:[https://github.com/m-abr/FCPCodebase](https://github.com/m-abr/FCPCodebase)

# FC Portugal Codebase <br> for RoboCup 3D Soccer Simulation League

![](https://s5.gifyu.com/images/Siov6.gif)

## About

The FC Portugal Codebase was mainly written in Python, with some C++ modules. It was created to simplify and speed up the development of a team for participating in the RoboCup 3D Soccer Simulation League. We hope this release helps existing teams transition to Python more easily, and provides new teams with a robust and modern foundation upon which they can build new features.


## Documentation

The documentation is available [here](https://docs.google.com/document/d/1aJhwK2iJtU-ri_2JOB8iYvxzbPskJ8kbk_4rb3IK3yc/edit)

## Features

- The team is ready to play!
    - Sample Agent - the active agent attempts to score with a kick, while the others maintain a basic formation
        - Launch team with: **start.sh**
    - Sample Agent supports [Fat Proxy](https://github.com/magmaOffenburg/magmaFatProxy) 
        - Launch team with: **start_fat_proxy.sh**
    - Sample Agent Penalty - a striker performs a basic kick and a goalkeeper dives to defend
        - Launch team with: **start_penalty.sh**
- Skills
    - Get Ups (latest version)
    - Walk (latest version)
    - Dribble v1 (version used in RoboCup 2022)
    - Step (skill-set-primitive used by Walk and Dribble)
    - Basic kick
    - Basic goalkeeper dive
- Features
    - Accurate localization based on probabilistic 6D pose estimation [algorithm](https://doi.org/10.1007/s10846-021-01385-3) and IMU
    - Automatic head orientation
    - Automatic communication with teammates to share location of all visible players and ball
    - Basics: common math ops, server communication, RoboViz drawings (with arrows and preset colors)
    - Behavior manager that internally resets skills when necessary
    - Bundle script to generate a binary and the corresponding start/kill scripts
    - C++ modules are automatically built into shared libraries when changes are detected
    - Central arguments specification for all scripts
    - Custom A* pathfinding implementation in C++, optimized for the soccer environment
    - Easy integration of neural-network-based behaviors
    - Integration with Open AI Gym to train models with reinforcement learning
        - User interface to train, retrain, test & export trained models
        - Common features from Stable Baselines were automated, added evaluation graphs in the terminal
        - Interactive FPS control during model testing, along with logging of statistics
    - Interactive demonstrations, tests and utilities showcasing key features of the team/agents
    - Inverse Kinematics
    - Multiple agents can be launched on a single thread, or one agent per thread
    - Predictor for rolling ball position and velocity
    - Relative/absolute position & orientation of every body part & joint through forward kinematics and vision
    - Sample train environments
    - User-friendly interface to check active arguments and launch utilities & gyms

## Citing the Project

```
@article{abreu2023designing,
  title={Designing a Skilled Soccer Team for RoboCup: Exploring Skill-Set-Primitives through Reinforcement Learning},
  author={Abreu, Miguel and Reis, Luis Paulo and Lau, Nuno},
  journal={arXiv preprint arXiv:2312.14360},
  year={2023}
}
```