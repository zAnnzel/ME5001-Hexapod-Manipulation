from Hebi import Hebi
from Hebi_Env import HebiEnv
import time

hebi = Hebi(real_robot_control=0, pybullet_on=1)
hebi.env.camerafollow = 1
time.sleep(2)
hebi.step(step_len=1, course=0, rotation=0, steps=2)
# hebi.step(step_len=1, course=0, rotation=90 , steps=2)
# hebi.step(step_len=1, course=0, rotation=0, steps=2)
hebi.stop()
time.sleep(60)