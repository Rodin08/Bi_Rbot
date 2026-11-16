# =============================================================================
# bi_rbot/tasks/locomotion/__init__.py  -  marks locomotion/ as a package
# =============================================================================
#
# JOB
#   Makes this folder importable. Can stay empty.
#
# WHAT THE LOCOMOTION TASK IS
#   ONE policy that follows three commands at the same time:
#       height    how high the body should be above the ground
#       speed     how fast to drive forward/backward (m/s)
#       yaw_rate  how fast to turn (rad/s)
#   Every episode gets random commands from the ranges in
#   locomotion_config.py, and they change in the middle of the episode, so the
#   policy learns all combinations AND switching between them.
#
# FILES IN THIS FOLDER
#   locomotion_config.py   command ranges, reward weights, curriculum stage
#   commands.py            picks random commands
#   locomotion_env.py      RobotEnv subclass: handles commands, adds them to obs
#   rewards.py             how well the commands are followed
#
# OPTIONAL
#   Re-export LocomotionEnv and LocomotionConfig here for shorter imports.
