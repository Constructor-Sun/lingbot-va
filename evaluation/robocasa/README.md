# RoboCasa Evaluation

This evaluator runs LingBot-VA against RoboCasa through the same websocket
server/client pattern used by the existing LIBERO and RoboTwin evaluators.

## Action Schema

RoboCasa actions in LingBot / VA use a single unified layout:

```text
0:3   end_effector_position
3:6   end_effector_rotation
6:7   gripper_close
7:11  base_motion
11:12 control_mode
```

`adapter.py` maps this flat action into the RoboCasa Gym wrapper action dict.

## Action Modes

Two action modes are supported via `ROBOCASA_ACTION_MODE`:

```text
full    : ee_pos(3) + ee_rot(3) + gripper(1) + base_motion(4) + control_mode(1)
no_base : ee_pos(3) + ee_rot(3) + gripper(1)
arm_only: alias of no_base
```

Default:

```text
ROBOCASA_ACTION_MODE=no_base
```

In `no_base` mode, the client automatically fills:

```text
action.base_motion = [0, 0, 0, 0]
action.control_mode = [0]
```

Use the same mode for both server and client:

```bash
ROBOCASA_ACTION_MODE=arm_only bash evaluation/robocasa/launch_server.sh
ROBOCASA_ACTION_MODE=arm_only bash evaluation/robocasa/launch_client.sh
```

So for `robocasa_no_base`, the actual LingBot output seen by the client is a
7D arm-only vector:

```text
[ee_pos(3), ee_rot(3), gripper(1)]
```

This layout must stay consistent with the RoboCasa LeRobot action export order
and the action stats written into `va_robocasa_cfg.py`.

## Norm Stats

To recompute target-demo action normalization stats and write them back to the
RoboCasa VA config:

```bash
python3 va/lingbot-va/wan_va/tools/compute_robocasa_norm_stat.py --write-config
```

The script defaults to:

```text
/data3/liu/exp/robocasa/datasets/v1.0/target/**/lerobot/meta/stats.json
```

Pass `--tasks TaskA TaskB` to restrict the pooled stats to selected tasks.

## Run

From `va/lingbot-va`:

```bash
bash evaluation/robocasa/launch_server.sh
```

In another shell:

```bash
bash evaluation/robocasa/launch_client.sh
```

Useful overrides:

```bash
PORT=29056 TEST_NUM=10 OUT_DIR=outputs/robocasa bash evaluation/robocasa/launch_client.sh OpenDrawer CloseFridge
```

By default, `launch_client.sh` loads the `guaranteed_no_base_motion` task group
from:

```text
evaluation/robocasa/task_mobility_groups.json
```

You can switch groups without editing code:

```bash
TASK_GROUP=likely_no_base_motion bash evaluation/robocasa/launch_client.sh
```
