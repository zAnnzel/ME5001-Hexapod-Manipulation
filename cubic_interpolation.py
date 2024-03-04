import numpy as np
import matplotlib.pyplot as plt


# Function to generate the cubic interpolate trajectory
def cubic_interpolate_path(initial_pos, end_pos, timelen):
    initial_v = np.array([0, 0, 0])
    end_v = np.array([0, 0, 0])
    a0 = initial_pos
    a1 = initial_v
    a2 = (3 / (timelen ** 2)) * (end_pos - initial_pos) - (1 / timelen) * (2 * initial_v + end_v)
    a3 = (2 / (timelen ** 3)) * (initial_pos - end_pos) + (1 / (timelen ** 2)) * (initial_v + end_v)  # 计算三次多项式系数
    traj = []
    time = []
    velocity = []
    acceleration = []
    for step in range(0, timelen*100, 1):
        t = step / 100
        q = np.array(a0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)).reshape(1, 3)  # 三次多项式插值的位置
        v = a1 + 2 * a2 * t + 3 * a3 * (t ** 2)  # 三次多项式插值的速度
        a = 2 * a2 + 6 * a3 * t  # 三次多项式插值的加速度
        traj.append(q)  # 保存位置、速度、加速度
        time.append(t)
        velocity.append(v)
        acceleration.append(a)
    traj = np.array(traj).reshape(-1, 3)
    time = np.array(time).reshape(-1, 1)
    return traj, time, velocity, acceleration


q0 = np.array([0, 0, 0])
q1 = np.array([100, 100, 100])
traj_, time_, velocity_, acceleration_ = cubic_interpolate_path(q0, q1, timelen=10)

fig = plt.figure()
fig.suptitle('Robotic Arm Path Interpolation in Joint Space')
ax = fig.add_subplot(221)
ax.plot(traj_[:, 0], traj_[:, 1])
ax.set_xlabel('Joint 1 Angle')
ax.set_ylabel('Joint 2 Angle')

bx = fig.add_subplot(222)
bx.plot(time_, traj_[:, 0])
bx.set_xlabel('t')
bx.set_ylabel('Joint 1 Angle')

cx = fig.add_subplot(223)
cx.plot(time_, velocity_)
cx.set_xlabel('t')
cx.set_ylabel('Velocity')

dx = fig.add_subplot(224)
dx.plot(time_, acceleration_)
dx.set_xlabel('t')
dx.set_ylabel('Acceleration')

plt.show()