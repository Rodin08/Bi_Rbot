# =============================================================================
# bi_rbot/base/__init__.py  -  marks base/ as a package
# =============================================================================
#
# JOB
#   A folder with an __init__.py is a "regular package": Python can import
#   from it with dotted paths like `bi_rbot.base.robot_env`.
#
# WHAT BASE/ IS
#   Everything that is the same for EVERY task: the robot, MuJoCo, the
#   step/reset loop, sensor reading, generic reward terms.
#   Rule: base/ never imports anything from tasks/. Tasks build on base,
#   never the other way round.
#
# WHAT CAN GO HERE (optional)
#   This file may stay empty. If you want shorter imports, re-export the
#   main classes here, so that
#       from bi_rbot.base import RobotEnv, RobotConfig
#   works instead of the longer
#       from bi_rbot.base.robot_env import RobotEnv
