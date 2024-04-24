import numpy as np
import matplotlib.pyplot as plt


# Function to generate the cubic interpolate trajectory
def cubic_interpolate_path(initial_pos, end_pos, timelen):
    initial_v = np.array([0, 0, 0])
    end_v = np.array([0, 0, 0])
    a0 = initial_pos
    a1 = initial_v
    a2 = (3 / (timelen ** 2)) * (end_pos - initial_pos) - (1 / timelen) * (2 * initial_v + end_v)
    a3 = (2 / (timelen ** 3)) * (initial_pos - end_pos) + (1 / (timelen ** 2)) * (initial_v + end_v)
    traj = []
    time = []
    velocity = []
    acceleration = []
    for step in range(0, timelen*100, 1):
        t = step / 100
        q = np.array(a0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)).reshape(1, 3)
        v = a1 + 2 * a2 * t + 3 * a3 * (t ** 2)
        a = 2 * a2 + 6 * a3 * t
        traj.append(q)
        time.append(t)
        velocity.append(v)
        acceleration.append(a)
    traj = np.array(traj).reshape(-1, 3)
    time = np.array(time).reshape(-1, 1)
    return traj, time, velocity, acceleration


# Function to generate the polynomial interpolate trajectory
def polynomial_interpolation_path(initial_pos, end_pos, timelen):
    #  Initial conditions
    initial_v = np.array([0, 0, 0])
    end_v = np.array([0, 0, 0])
    initial_a = np.array([0, 0, 0])
    end_a = np.array([0, 0, 0])
    a0 = initial_pos
    a1 = initial_v
    a2 = initial_a / 2
    a3 = (20 * (end_pos - initial_pos) - (8 * end_v + 12 * initial_v) * timelen - (3 * initial_a - end_a) * (timelen ** 2)) / (2 * (timelen ** 3))
    a4 = (30 * (initial_pos - end_pos) + (14 * end_v + 16 * initial_v) * timelen + (3 * initial_a - 2 * end_a) * (timelen ** 2)) / (2 * (timelen ** 4))
    a5 = (12 * (end_pos - initial_pos) - (6 * (end_v + initial_v)) * timelen - (initial_a - end_a) * (timelen ** 2)) / (2 * (timelen ** 5))
    traj = []
    time = []
    velocity = []
    acceleration = []
    for step in range(0, timelen*100, 1):
        t = step / 100
        q = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
        v = a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
        a = 2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
        traj.append(q)
        time.append(t)
        velocity.append(v)
        acceleration.append(a)
    traj = np.array(traj).reshape(-1, 3)
    time = np.array(time).reshape(-1, 1)
    return traj, time, velocity, acceleration


# q0 = np.array([-0.49188775, 0.15800083,  0.01668477])
# q1 = np.array([-0.53168775, -2.82699917, -2.37131523])
# traj_, time_, velocity_, acceleration_ = cubic_interpolate_path(q0, q1, timelen=1)

q0_array = np.array([-0.49188775, 0.15800083,  0.01668477])
q1_array = np.array([-0.53168775, -2.82699917, -2.37131523])
traj_2, time_2, velocity_2, acceleration_2 = polynomial_interpolation_path(q0_array, q1_array, timelen=1)

# # Plot the cubic interpolate trajectory
# fig = plt.figure()
# fig.suptitle('Robotic Arm Path Interpolation in Joint Space')
# ax = fig.add_subplot(221, projection='3d')
# ax.plot(traj_[:, 0], traj_[:, 1])
# ax.set_xlabel('Joint 1 Angle')
# ax.set_ylabel('Joint 2 Angle')
# ax.set_zlabel('Joint 3 Angle')
#
# bx = fig.add_subplot(222)
# bx.plot(time_, traj_[:, 0])
# bx.set_xlabel('t')
# bx.set_ylabel('Joint 1 Angle')
#
# cx = fig.add_subplot(223)
# cx.plot(time_, velocity_)
# cx.set_xlabel('t')
# cx.set_ylabel('Velocity')
#
# dx = fig.add_subplot(224)
# dx.plot(time_, acceleration_)
# dx.set_xlabel('t')
# dx.set_ylabel('Acceleration')
#
# plt.show()


# Plot the polynomial interpolate trajectory
fig2 = plt.figure(figsize=(8,8))
fig2.subplots_adjust(hspace=0.5, wspace=0.5)
fig2.suptitle('Robotic Arm Path Interpolation in Joint Space')
ax = fig2.add_subplot(221, projection='3d')
# fig2.tight_layout(h_pad=3)
ax.plot(traj_2[:, 0], traj_2[:, 1])
ax.set_xlabel('Joint 1 Angle')
ax.set_ylabel('Joint 2 Angle')
ax.set_zlabel('Joint 3 Angle')

bx = fig2.add_subplot(222)
bx.plot(time_2, traj_2)
bx.set_xlabel('t')
bx.set_ylabel('Joint Angle')

cx = fig2.add_subplot(223)
cx.plot(time_2, velocity_2)
cx.set_xlabel('t')
cx.set_ylabel('Velocity')

dx = fig2.add_subplot(224)
dx.plot(time_2, acceleration_2)
dx.set_xlabel('t')
dx.set_ylabel('Acceleration')

plt.show()