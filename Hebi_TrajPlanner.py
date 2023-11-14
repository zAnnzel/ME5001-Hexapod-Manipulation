import numpy as np
import hebi
from functions import solveIK, rot, trans
# default robot leg end-effecter position w.r.t body frame
eePos = np.array([[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                  [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                  [-0.2249,   -0.2249,    -0.2249,    -0.2249,    -0.2249,    -0.2249]])
class TrajPlanner:
    def __init__(self, neutralPos=eePos):
        self.dutyfactor = 0.5
        self.period = 1.0
        self.dt = 1/240 # pybullet default
        self.eePos = neutralPos # neutral position for the robot
        self.eeAng = np.array([0., 0., 0., 0., 0., 0.,]) # the diviation of each leg from neutral position, use 0. to initiate a float type array
        self.array_dim = int(np.around(self.period / self.dt)) # timesteps of a complete period
        self.stance_dim = int(np.around(self.period * self.dutyfactor / self.dt)) # 120
        self.swing_dim = self.array_dim - self.stance_dim # 120
        self.traj_dim = np.maximum(self.swing_dim, self.stance_dim)
        self.clearance = 0.1 # maximum foot clearance when the robot lifts its leg
        self.tripod1 = [0 ,3, 4] # leg index for leg 1, 4, 5
        self.tripod2 = [1, 2, 5] # leg index for leg 2, 3, 6

    def get_loco_traj(self, init_pose, step_len, course, rotation, flag, timestep):
        '''
        Compute the leg trajectories of all six legs within a stride
        :param init_pose: initial pose of all six legs' task coordinate frame
        :param step_len: step length, translational displacement of the leg's task coordinate frame with respect to the body frame
        :param course: course angle in radians
        :param rotation: rotation angle, rotational displacement of the leg's task coordinate frame with respect to the body frame
        :param flag: use the parity of the flag to determine which set of tripod to move forward, odd flag for tripod1, even flag for tripod2
        :param timestep: current timestep
        :return traj: position of all six legs at a given timestep of a stride trajectory
        :return end_pose: desired end pose of this trajectory of all six legs' task coordinate frame
        '''
        traj = np.zeros((3, 6))
        end_pose = self._get_end_pose(step_len, course, rotation, flag)
        curve_type = ['swing', 'stance', 'stance', 'swing', 'swing', 'stance'] if flag % 2 == 0 else ['stance', 'swing', 'swing', 'stance', 'stance' , 'swing']

        for leg_index in range(6):
            traj[:,leg_index] = self._compute_traj(init_pose[:,leg_index], end_pose[:,leg_index], curve_type[leg_index], leg_index, timestep)

        return traj, end_pose

    def _get_end_pose(self, step_len, course, rotation, flag=0): # use flag to determine which set of tripod to move forward
        '''
        Compute the end pose of each leg's task coordinate frame
        :param step_len: step length, translational displacement of the leg's task coordinate frame with respect to the body frame
        :param course: course angle in radians
        :param rotation: rotation angle, rotational displacement of the leg's task coordinate frame with respect to the body frame
        :param flag: use the parity of the flag to determine which set of tripod to move forward, odd flag for tripod1, even flag for tripod2
        :return end_pose: end pose of all six legs' task coordinate frame
        '''
        end_pos = np.zeros((3, 6))
        end_ang = np.zeros((6,))
        neutral_pose = np.zeros((4,1))
        if flag % 2==0: # even flag for tripod2, odd flag for tripod1
            for leg_index in self.tripod1:
                end_pos[:, leg_index] = np.reshape(trans(neutral_pose[:3], +step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] + rotation / 2
            for leg_index in self.tripod2:
                end_pos[:, leg_index] = np.reshape(trans(neutral_pose[:3], -step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] - rotation / 2
        else:
            for leg_index in self.tripod1:
                end_pos[:, leg_index] = np.reshape(trans(neutral_pose[:3], -step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] - rotation / 2
            for leg_index in self.tripod2:
                end_pos[:, leg_index] = np.reshape(trans(neutral_pose[:3], +step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] + rotation / 2

        end_pose = np.vstack((end_pos, end_ang)) # np.vstack-按垂直方向（行顺序）堆叠数组构成一个新的数组
        return end_pose

    def pose2pos(self, pose, leg_index):
        '''
        Compute the position of each leg from the pose of leg's task coordinate frame
        :param pose: 4x1 array, pose of leg's task coordinate frame with respect to body frame
        :param leg_index: int, index of leg
        :return pos: position of leg
        '''
        pos = rot(pos=self.eePos[:,leg_index]+pose[:3], angle=pose[3], pivot=pose[:3])
        return pos

    def _compute_traj(self, init_pose, end_pose, curve_type, leg_index, timestep):  # pose of single leg
        '''
        Compute the foot trajectory for a single leg within a stride, and return the position of the foot at a given timestep
        The trajectory planner uses a cycloid curve to generate the foot trajectory, you can use plot_traj.py to visualise this curve
        The cycloid parametric equation is:
            x = r * (t - sin(t))
            y = r * (1 - cos(t))
        The height (y_max) of this cycloid is 2*r and length (x_max) is 2*pi*r

        If there is a point rigidly attached to the centre of this circle and the distance is R, the trajectory of this point is:
            x = r * (t - R * sin(t))
            y = r * (1 - R * cos(t))

        Suppose the velocity of circle centre is v0, the angular velocity of the circle is w0, we have:
            v0 = w0 * r
        The points with the same x coordinates with circle centre have the velocity of:
            v = v0 + w0 * (y - yc) = v0 - w0 * R
        where y is the y coodinate of the point, yc is the y coordinate of the circle centre
        If R = 0, v is the velocity of circle centre and equals to v0, 
        if R = r, v is the velocity of the point contacting the ground, which is 0
        Our desired v for the foot trajectory is -v0, and the corresponding R = 2

        If we want to rescale it to use as a foot trajectory, then the resized equation is:
            x = s * (t - 2 * sin(t)) / (2 * pi)
            y = c * (1 - 2 * cos(t) + 1) / 4
        where s stands for step length and c stands for foot clearance

        :param init_pose: initial pose of the each leg'g task coordinate with respect to the body frame
        :param end_pose: end pose of the each leg'g task coordinate with respect to the body frame
        :param curve_type: 'swing' or 'stance'
        :param leg_index: index of the leg
        :param timestep: correponding timestep of the desired waypoint within a stride trajectory
        :return: position of the foot with respect to the body frame at the given timestep
        '''
        traj = np.zeros((self.swing_dim, 3))
        deltaPose = end_pose - init_pose

        if curve_type == 'swing':
            dt = 2 * np.pi / self.swing_dim
            clearance = np.array((0, 0, self.clearance))  # foot clearance
            t = timestep * dt
            pose = init_pose + deltaPose * (t - 2 * np.sin(t)) / ( 2 * np.pi)
            traj = self.pose2pos(pose, leg_index) + clearance * (1 - 2 * np.cos(t) + 1) / 4
            print(pose)
            print(traj)
        elif curve_type == 'stance':
            dPose = deltaPose / self.traj_dim
            pose = timestep * dPose + init_pose
            traj = self.pose2pos(pose, leg_index)
        else:
            raise ValueError('Wrong curve type, the available types are: \'swing\' or \'stance\'.')

        return traj

    def general_traj(self, waypoints, total_time=1, time_vector=[]):
        '''
        Use Hebi's trajectory interpolation function to generate a series of more dense interpolated trajectories in jointspace
        :param waypoints: a list of waypoints, each waypoint can be either a 3x6 matrix (workspace command) or a 18x1 vector (jointspace command)
        :param total_time: the total time of robot following the trajectory
        :param time_vector: a list of time points, each time point is the time when the robot reaches the corresponding waypoint, therefore the length of time_vector should be the same as the length of waypoints
        :return: a list of interpolated trajectories in jointspace
        '''
        num_pos = len(waypoints) # the number of given positions in a trajectory
        jointspace_command = np.zeros(shape=(18,num_pos))
        # unify all waypoints to joint space
        for i in range(num_pos):
            if np.shape(waypoints[i]) == (3,6): # input workspace command as waypoint
                _, jointspace_command[:,i] = solveIK(waypoints[i]) # discard the first output which is for pybullet environment
            elif np.shape(waypoints[i]) == (18,): # input jointspace command as waypoint
                jointspace_command[:,i] = waypoints[i]
            else:
                raise ValueError('Command that Yuna cannot recognise')
        # default time vector, assuming the time is evenly distributed, otherwise please assign customised time vector in argument
        if not time_vector:
            interval = total_time / (num_pos - 1) # 间隔时间
            time_vector = [interval * _  for _ in range(num_pos)] # 生成时间向量
        # create trajectory
        trajectory = hebi.trajectory.create_trajectory(time_vector, jointspace_command)
        duration = trajectory.duration # _end_time - _start_time
        len_traj = int(duration / self.dt) + 1
        traj = np.zeros((len_traj,18))
        for i in range(len_traj):
            pos, vel, acc = trajectory.get_state(i*self.dt)
            traj[i] = pos
        return traj

    def front_leg_workspace_traj(self, step, leg_index):
        init_point = np.array([[0.85, 0.85, 0.23, 0.23, -0.4, -0.4],
                              [0.1, -0.1, 0.4, -0.4, 0.4, -0.4],
                              [0.1, 0.1, -0.12, -0.12, -0.12, -0.12]])
        end_point = np.array([[0.2, 0.2, 0.23, 0.23, -0.4, -0.4],
                              [0.1, -0.1, 0.4, -0.4, 0.4, -0.4],
                              [0.5, 0.5, -0.12, -0.12, -0.12, -0.12]])
        dt = 2 / 100
        t = step * dt
        traj = init_point[:, leg_index] + [(1.414 / 4 * np.cos(t) - 1.414 / 4), 0, (1 / 4 * np.sin(t))]
        return traj

    def front_leg_jointspace_traj(self, step, leg_index):
        # init_point = np.array([[0.51589, 0.51589, 0.0575, 0.0575, -0.45839, -0.45839],
        #                        [0.23145, -0.23145, 0.5125, -0.5125, 0.33105, -0.33105],
        #                        [-0.12, -0.12, -0.12, -0.12, -0.12, -0.12]])
        # init_point = np.array([-0.46336853, -0.15668338, -0.31367825, # leg 0
        #                         -0.64025334, -0.32865358, -1.89880505,
        #                         -0.27359199, -0.32446194, -1.80610188,
        #                         0.46336853,  0.15668338,  0.31367825, #leg 1
        #                         0.64025296,  0.3286524,   1.89880162,
        # #                         0.27359199,  0.32446194,  1.80610188])
        # end_point = np.array([-0.49474198, -0.67771703, -4.46762366, # leg 0
        #                       -0.64025334, -0.32865358, -1.89880505,
        #                       -0.27359199, -0.32446194, -1.80610188,
        #                       0.49474198,  0.67771703,  4.46762366, # leg 1
        #                       0.64025296,  0.3286524,   1.89880162,
        #                       0.27359199, 0.32446194,  1.80610188])
        # init_point = np.array([-0.46336853, -0.15668338, -0.31367825, # leg 0
        #                         0.46336853,  0.15668338,  0.31367825, # leg 1
        #                        -0.64025334, -0.32865358, -1.89880505,
        #                         0.64025296,  0.3286524,   1.89880162,
        #                        -0.27359199, -0.32446194, -1.80610188,
        #                         0.27359199,  0.32446194,  1.80610188])
        init_point = np.array([-0.49188775, 0.15800083,  0.01668477,
         0.49188775, - 0.15800083, - 0.01668477,
         - 0.64025334, - 0.32865358, - 1.89880505,
         0.64025296,  0.3286524,   1.89880162,
         - 0.27359199, - 0.32446194, - 1.80610188,
         0.27359199,  0.32446194,  1.80610188])
        end_point = np.array([-0.49474198, -0.67771703, -4.46762366, # leg 0
                               0.49474198,  0.67771703,  4.46762366, # leg 1
                               -0.64025334, -0.32865358, -1.89880505,
                               0.64025296,  0.3286524,   1.89880162,
                               -0.27359199, -0.32446194, -1.80610188,
                               0.27359199,  0.32446194,  1.80610188])

        # deltapos = (end_point - init_point)[:, leg_index]
        dt = 1 / 100  # 1 / 100
        t = step * dt
        traj = init_point[(leg_index * 3): (leg_index * 3 + 3)] + [(-1 + leg_index * 2) * 0.01 * t, (-1 + leg_index * 2) * 0.75 * t, (-1 + leg_index * 2) * 0.6 * t]
        return traj

if __name__ == '__main__':
    # plot trajectory of a leg
    import numpy as np
    import matplotlib.pyplot as plt
    # params
    tp = TrajPlanner()
    step_len = 0.1
    course = 0
    rotation = 0
    leg_index = 3
    # compute trajectory
    end_pose = tp._get_end_pose(step_len, course, rotation)[:, leg_index]
    init_pose = - end_pose
    traj_swing = np.zeros((tp.swing_dim, 3))
    traj_stance = np.zeros((tp.stance_dim, 3))
    for timestep in range(tp.traj_dim):
        traj_swing[timestep] = tp._compute_traj(init_pose, end_pose, curve_type='swing', leg_index=leg_index, timestep=timestep)
        traj_stance[timestep] = tp._compute_traj(end_pose, init_pose, curve_type='stance', leg_index=leg_index, timestep=timestep)
    traj = np.vstack((traj_swing, traj_stance))
    # plot
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, s=1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_aspect('equal', 'box')
    plt.show()
    pass