# =============================================================================
# bi_rbot/__init__.py  -  package entry point + task registration
# =============================================================================
#
# JOB
#   Python runs this file automatically the first time anything imports
#   bi_rbot (or something inside it). Its one job: REGISTER every task with
#   Gymnasium, so the rest of the project creates environments by NAME
#   instead of importing classes directly:
#
#       env = gym.make("BiRBot-Locomotion-v0")
#
# WHAT GOES HERE
#   One gym.register(...) call per task. Each call links three things:
#       id          -> the public name, e.g. "BiRBot-Locomotion-v0"
#                      (the "-v0" is a version; bump it when you change the
#                      observations, so old models aren't loaded by mistake)
#       entry_point -> where the env class lives, written as a string
#                      "module.path:ClassName", e.g.
#                      "bi_rbot.tasks.locomotion.locomotion_env:LocomotionEnv"
#       kwargs      -> optional arguments passed to the class, e.g. a config
#
#   Nothing else. No env logic, no numbers.
#
# HOW IT CONNECTS
#   train.py / watch.py
#       --gym.make("<id>")-->  this registration
#       --entry_point------->  tasks/<task>/<task>_env.py  -> the env class
#
#   train.py will later take `--task locomotion` and turn it into the id.
#
# WATCH OUT
#   SubprocVecEnv starts fresh Python processes that have NOT imported
#   bi_rbot, so they don't know about this registration. Use the id with a
#   module prefix:  "bi_rbot:BiRBot-Locomotion-v0"
#   Gymnasium then imports bi_rbot (= runs this file) in each process first.
#
# WHEN YOU ADD A TASK
#   Create tasks/<new_task>/ and add one more gym.register(...) here.
