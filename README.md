# VLA data collecting via RL

## 概述

本项目用作基于 Isaac Lab 构建项目。旨在实现利用强化学习PPO算法针对VLA的仿真环境进行数据采集工作

## 启动训练

针对所需要的任务，进行训练，得到策略后去产生数据

当前任务：stack blue cube on the red cube(把蓝色方块放到红色方块上)

运行以下指令前确保在正确工作空间下且已经激活ISAAC LAB的虚拟环境

```bash
cd vla_data_collect_from_rl/
source /home/chiro/IsaacLab/env_isaaclab/bin/activate
```

### 第一次运行

```bash
python scripts/rsl_rl/train.py --task Franka-VLA-Stack-v0 --num_envs 16 --enable_cameras --max_iterations 90000
```

--task 任务名
--num_envs 并行环境个数
--enable_cameras 启动cam渲染
--max_iterations 总迭代轮次

### 断点续练

```bash
python scripts/rsl_rl/train.py --task Franka-VLA-Stack-v0 --num_envs 16 --enable_cameras --max_iterations 90000 --resume --load_run 2026-04-26_23-02-08 --checkpoint model_200.pt
```

### 演示
```bash
python scripts/rsl_rl/play.py --task Franka-VLA-Stack-v0 --num_envs 1 --checkpoint /home/chiro/vla_data_collect_from_rl/vla_data_collect_from_rl/logs/rsl_rl/franka_vla_stack_expert_16envs/2026-04-26_23-02-08/model_200.pt --enable_cameras
```

--resume 确保在指定检查点开始训练而不是从头
--load_run 检查点所在日期文件名
--checkpoint 检查点文件名

tips：检查点文件位于`/home/chiro/vla_data_collect_from_rl/vla_data_collect_from_rl/logs/rsl_rl/franka_vla_stack_expert_16envs/2026-04-26_23-02-08`下的`model_200.pt`

## 安装（官方教程）

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/vla_data_collect_from_rl

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/vla_data_collect_from_rl/vla_data_collect_from_rl/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/vla_data_collect_from_rl"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```


python scripts/rsl_rl/train.py --task Franka-VLA-Stack-v0 --num_envs 16 --enable_cameras --max_iterations 90000




################################################################################
                          Learning iteration 318/6000                            

                            Total steps: 5226496 
                       Steps per second: 132 
                        Collection time: 123.369s 
                          Learning time: 0.058s 
                        Mean value loss: 0.1962
                    Mean surrogate loss: -0.0061
                      Mean entropy loss: 9.9086
                            Mean reward: 35.50
                    Mean episode length: 600.00
                        Mean action std: 0.85
                   Episode_Reward/alive: 0.1000
     Episode_Reward/penalty_wrong_grasp: 0.0000
    Episode_Reward/approach_target_cube: 0.7150
       Episode_Reward/grasp_target_cube: 1.1539
        Episode_Reward/lift_target_cube: 0.5047
    Episode_Reward/align_with_base_cube: 0.8496
           Episode_Reward/stack_success: 0.0000
             Episode_Reward/action_rate: -0.1438
           Episode_Termination/time_out: 1.0000
            Episode_Termination/success: 0.0000
 Episode_Termination/cube_out_of_bounds: 0.0000
--------------------------------------------------------------------------------
                         Iteration time: 123.43s
                           Time elapsed: 10:37:33
                                    ETA: 21:14:16



python scripts/rsl_rl/play.py --task Franka-VLA-Stack-v0 --num_envs 1 --checkpoint /home/chiro/vla_data_collect_from_rl/vla_data_collect_from_rl/logs/rsl_rl/franka_vla_stack_expert_16envs/2026-04-26_23-02-08/model_200.pt --enable_cameras


python scripts/rsl_rl/train.py --task Franka-VLA-Stack-v0 --num_envs 16 --enable_cameras --resume --load_run 2026-04-26_23-02-08 --checkpoint model_200.pt