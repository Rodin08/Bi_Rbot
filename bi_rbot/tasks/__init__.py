# =============================================================================
# bi_rbot/tasks/__init__.py  -  marks tasks/ as a package
# =============================================================================
#
# JOB
#   Makes tasks/ importable (bi_rbot.tasks.locomotion...). Can stay empty.
#
# WHAT TASKS/ IS
#   One folder per task = per kind of behaviour you train a SEPARATE policy
#   for. Each task folder has the same files:
#       <task>_config.py   what makes this task different (data only)
#       <task>_env.py      small RobotEnv subclass that overrides hooks
#       rewards.py         reward terms only this task needs
#
# TASKS
#   locomotion/   drive + turn + height, all in ONE policy via commands
#
#   Possible later:
#   recovery/     get up after falling (different start state, no fall end)
#   rough/        locomotion on uneven ground
#
#   Different height/speed values are NOT separate tasks. They are commands
#   inside locomotion.
#
# RULES
#   - Every task env inherits DIRECTLY from RobotEnv (no task inherits from
#     another task) -> max. two levels, easy to follow.
#   - Tasks never override step() or reset(), only the hooks.
#   - Register every task in bi_rbot/__init__.py.
