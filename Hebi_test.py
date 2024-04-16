from Hebi_TrajPlanner import TrajPlanner
from Hebi_grasp import Grasper
import pybullet as p
from Hebi_Env import HebiEnv
import numpy as np
from functions import trans, solveIK, solveFK
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time
import imageio
import wandb
import math

# wandb.init(project='Hebi-test', name='force-torque')

init_leg = np.array([0.51589, 0.51589, 0.0575, 0.0575, -0.45839, -0.45839,
                    0.23145, -0.23145, 0.5125, -0.5125, 0.33105, -0.33105,
                    -0.15, -0.15, -0.15, -0.15, -0.15, -0.15])

class Hebi:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=False, pybullet_on=True):
        # initialise the environment
        self.env = HebiEnv(visualiser=visualiser, camerafollow=camerafollow, real_robot_control=real_robot_control, pybullet_on=pybullet_on)
        self.eePos = self.env.eePos.copy()
        self.eeAng = np.array([0., 0., 0., 0., 0., 0.,]) # the diviation of each leg from neutral position, use 0. to initiate a float type array
        self.init_pose = np.zeros((4, 6))
        self.current_pose = np.copy(self.init_pose)

        self.real_robot_control = real_robot_control
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.env.xmk, self.env.imu, self.env.hexapod, self.env.fbk_imu, self.env.fbk_hp, self.env.group_command, self.env.group_feedback
        
        self.trajplanner = TrajPlanner(neutralPos=self.eePos)
        self.grasper = Grasper(neutralPos=self.eePos)

        self.max_step_len = 0.2 # maximum stride length in metre 最大步幅
        self.max_rotation = 20 # maximum turn angle in degrees 最大转弯角度
        # set default value to step parameters, they may change using get_step_params() function
        self.step_len = 0. # stride length in metre 步幅
        self.course = 0. # course angle in degrees 航向角
        self.rotation = 0. # turn angle in degrees 转弯角度
        self._step_len = np.copy(self.step_len) # record the last step length
        self._course = np.copy(self.course) # record the last step course
        self._rotation = np.copy(self.rotation) # record the last step rotation
        self.cmd_step_len = np.copy(self.step_len)
        self.cmd_course = np.copy(self.course)
        self.cmd_rotation = np.copy(self.rotation)
        self.cmd_steps = 1 # number of steps to take

        self.traj_dim = self.trajplanner.traj_dim # trajectory dimension for walking and turning, they both have same lenghth
        self.flag = 0 # a flag to record how many steps achieved 计步
        self.is_moving = False # Ture for moving and False for static
        self.smoothing = True # Set to True to enable step smoothing

        self.leg_index = 0


    def step(self, *args, **kwargs):
        '''
        The function to enable Yuna robot step one stride forward, the parameters are get from get_step_params() or manually set
        :param step_len: The step length the robot legs cover during its swing or stance phase in metres, this is measured under robot body frame. The actual step length of first step is halved
        :param course: The robot moving direction, this is measured under robot body frame
        :param rotation: The rotation of robot body per step in radians. The actual rotation of first step is halved
        :param steps: The number of steps the robot will take
        :return: if the step is executed, return True, else return False
        '''
        self.get_step_params(*args, **kwargs) #前进 *args-可变参数列表 **kwargs-关键字参数
        cmd_steps = self.cmd_steps
        self.cmd_steps = 1

        if self.cmd_step_len == 0.0 and self.cmd_rotation == 0.0 and np.equal(self.current_pose, self.init_pose).all(): #np.equal-比较两个数组是否相等
            self.is_moving = False
            return False
        else:
            # change robot status and start moving
            self.is_moving = True
            if self.cmd_step_len == 0 and self.cmd_rotation == 0: # robot executes a single step to stop
                cmd_steps = 1
                self.is_moving = False
            for step in range(int(cmd_steps)):
                for i in range(self.traj_dim):
                    self._smooth_step()
                    traj, end_pose = self.trajplanner.get_loco_traj(self.current_pose, self.step_len, self.course, self.rotation, self.flag, i)
                    self.env.step(traj)
                self.current_pose = end_pose
                self.flag += 1
            return True
        
    def goto(self, dx, dy, dtheta):
        '''
        Function to move Yuna robot to a specific position
        :param dx: The x-axis displacement in metres
        :param dy: The y-axis displacement in metres
        :param dtheta: The rotation of robot body orientation in degrees
        '''
        self.smoothing = False # disable step smoothing for more accurate positioning
        cor_coeff_disp = 1.075
        cor_coeff_ang = 1.053
        dx, dy, dtheta = dx*cor_coeff_disp, dy*cor_coeff_disp, dtheta*cor_coeff_ang
        lin_disp = np.sqrt(dx**2 + dy**2)
        ang_disp = np.abs(dtheta)
        lin_steps = np.ceil(lin_disp / self.max_step_len)
        ang_steps = np.ceil(ang_disp / self.max_rotation)
        steps = np.int64(np.max((lin_steps, ang_steps)))

        step_len = lin_disp / steps
        course = np.rad2deg(np.arctan2(dy, dx))
        rotation = np.sign(dtheta) * ang_disp / steps

        for step in range(steps):
            self.step(step_len=step_len, course=course, rotation=rotation)
            course -= rotation
            # if step == 0:
            #     course -= rotation/2
            # else:
            #     course -= rotation

        self.stop()
        self.smoothing = True

    def stop(self):
        '''
        The function to stop Yuna's movements and reset Yuna's pose
        :return: None
        '''
        if self.is_moving:
            self.step(step_len=0,rotation=0)
            self.is_moving = False
    
    def disconnect(self):
        '''
        Disable real robot motors, disconnect from pybullet environment and exit the programme
        :return: None
        '''
        self.env.close()

    def get_step_params(self, *args, **kwargs):
        '''
        This function is used to listen to the commands from the user
        :param step_len: The step length the robot legs cover during its swing or stance phase, this is measured under robot body frame. The actual step length of first step is halved
        :param course: The robot moving direction, this is measured under robot body frame
        :param rotation: The rotation of robot body per step. The actual rotation of first step is halved
        :param steps: The number of steps the robot will take
        :return: None
        '''
        if len(args) > 0 or len(kwargs) > 0:# if there is any input command
            if len(args) + len(kwargs) > 4:
                raise ValueError('Expected at most 4 arguments: step_len, course, rotation and steps, got %d' % (len(args) + len(kwargs)))
            # set default values
            legal_keys = ['step_len', 'course', 'rotation', 'steps'] #设置指令参数
            default_values = [0., 0., 0., 1] #设置指令参数的默认值
            for key in legal_keys:
                setattr(self, key, default_values[legal_keys.index(key)])
            # combine the args and kwargs, if there is a value in args not stated in kwargs, it will be assigned to the first key in legal_keys that is not in kwargs
            for value in args:
                for key in legal_keys:
                    if key not in kwargs:
                        kwargs[key] = value
                        break
            # check the input commands and set input parameters as attributes
            for key, value in kwargs.items():
                if key not in legal_keys:
                    raise TypeError('Invalid keyword argument: {}'.format(key))
                # pre-processing of the input commands
                if key == 'step_len':
                    value = np.clip(value, -self.max_step_len, self.max_step_len) #np.clip()将数组中的元素限制在a_min, a_max之间
                if key == 'course':
                    value = np.deg2rad(value) #np.deg2rad()将角度转换为弧度
                if key == 'rotation':
                    value = np.clip(value, -self.max_rotation, self.max_rotation)
                    value = np.deg2rad(value)
                if key == 'steps':
                    value = np.ceil(np.abs(value)) #np.ceil()返回大于或者等于指定表达式的最小整数 ; np.abs()返回数字的绝对值
                setattr(self, 'cmd_' + key, value)

    def _smooth_step(self):
        '''
        This function is used to smooth the robot's movements to avoid abrupt changes in robot's legs' task coordinate pose
        :return: None
        '''
        rho = 0.05# soft copy rate
        if self.smoothing:
            _pos = trans((0.,0.), self._step_len, self._course)
            cmd_pos = trans((0.,0.), self.cmd_step_len, self.cmd_course)
            dpos = np.linalg.norm(cmd_pos - _pos) #np.linalg.norm()求2范数
            print(dpos)
            _rot = self._rotation
            cmd_rot = self.cmd_rotation
            drot = np.rad2deg(np.abs(cmd_rot - _rot))
            if dpos < 2 * self.max_step_len / 10  and drot < 2 * self.max_rotation / 10 and self.cmd_step_len == 0 and self.cmd_rotation == 0:
                rho = 1
            pos = rho * cmd_pos + (1 - rho) * _pos
            self.step_len = np.sqrt(pos[0]**2 + pos[1]**2)
            self.course = np.arctan2(pos[1], pos[0])
            self.rotation = rho * self.cmd_rotation + (1 - rho) * self._rotation

            self._step_len = np.copy(self.step_len)
            self._course = np.copy(self.course)
            self._rotation = np.copy(self.rotation)
        else:
            self.step_len = self.cmd_step_len
            self.course = self.cmd_course
            self.rotation = self.cmd_rotation
    
    def _get_current_pos(self):
        '''
        Get the current end effector positions of all 6 legs from the current pose
        :return: current leg end effector positions based on the current pose of each leg's task coordinate frame with respect to the robot body frame
        '''
        current_pos = np.zeros((3, 6))
        for leg_index in range(6):
            current_pos[:, leg_index] = self.trajplanner.pose2pos(self.current_pose[:, leg_index], leg_index)
        return current_pos


    ## hexapod抓取物体
    # 使用逆运动学让机器人的单条腿移动到指定位置
    def move_leg(self, waypoints, time_vector, total_time):
        self.is_moving = True
        self.time_vector = time_vector
        self.total_time = total_time
        traj = self.trajplanner.general_traj(waypoints, total_time=1, time_vector=[])
        positions = []
        for i in range(len(traj)):
            traj_ = np.array(traj[i])
            traj_ = traj_.reshape(3,6)
            self.env.step(traj_)
            position, velocity, torque = self.env.step(traj_)
            positions.append(position)
        return positions

    # 调整支撑腿位置
    def adjust_leg(self, leg_4_pos, leg_5_pos):
        leg_clearance = -0.05
        time_vector_1 = [0, 0.5, 1]

        # leg 3, 4
        init_pos = init_leg
        middle_pos = np.hstack((init_pos[0:2], (init_pos[2] + leg_4_pos[0]) / 2, (init_pos[2] + leg_4_pos[0]) / 2,
                                init_pos[4:8], (init_pos[8] + leg_4_pos[1]) / 2, (-1) * (init_pos[8] + leg_4_pos[1]) / 2,
                                init_pos[10:14], leg_clearance, leg_clearance, init_pos[16:18]))
        end_pos = np.hstack((init_pos[0:2], leg_4_pos[0], leg_4_pos[0], init_pos[4:8],
                            leg_4_pos[1], (-1) * leg_4_pos[1], init_pos[10:18]))
        waypoints_34 = np.vstack((init_pos, middle_pos, end_pos))
        self.move_leg(waypoints_34, time_vector_1, total_time=1)

        # leg 5
        init_pos = end_pos
        middle_pos_2 = np.hstack((init_pos[0:4], (init_pos[4] + leg_5_pos[0]) / 2,
                                  init_pos[5:10], (init_pos[10] + leg_5_pos[1]) / 2,
                                  init_pos[11:16], leg_clearance, init_pos[17]))
        end_pos_2 = np.hstack((init_pos[0:4], leg_5_pos[0], init_pos[5:10], leg_5_pos[1], init_pos[11:18]))
        waypoints_5 = np.vstack((init_pos, middle_pos_2, end_pos_2))
        self.move_leg(waypoints_5, time_vector_1, total_time=1)

        # leg 6
        init_pos = end_pos_2
        middle_pos_3 = np.hstack((init_pos[0:5], (init_pos[5] + leg_5_pos[0]) / 2,
                                  init_pos[6:11], (init_pos[11] + (-1) * leg_5_pos[1]) / 2,
                                  init_pos[12:17], leg_clearance))
        end_pos_3 = np.hstack((init_pos[0:5], leg_5_pos[0], init_pos[6:11], (-1) * leg_5_pos[1], init_pos[12:18]))
        waypoints_6 = np.vstack((init_pos, middle_pos_3, end_pos_3))
        self.move_leg(waypoints_6, time_vector_1, total_time=1)
        print('leg 6', end_pos_3)
        return end_pos_3

    # 调整base姿态
    def adjust_base(self, lift_pos, angle):
        self.is_moving = True
        timestep = round(math.radians(angle) / 0.005)  # 0.01
        lift_pos = np.array(lift_pos)
        _, pos = solveIK(lift_pos.reshape(3, 6))
        init_point = np.array(pos)
        base_traj = []

        for step in range(timestep):
            traj_ = init_point + np.array([0, 0, 0, 0, 0, 0,
                                           0, (step * 0.0025), (step * 0.0025), 0, (step * -0.0025), (step * -0.0025),
                                           0, (step * -0.005), (step * -0.0025), 0, (step * 0.005), (step * 0.0025)])
            base_traj.append(traj_)
            self.env.step(traj_)
        print('base', base_traj[-1])
        return base_traj[-1]

    # 用逆运动学计算俯仰角
    def adjust_base_ik(self, lift_pos, angle):
        self.is_moving = True
        timestep = round(math.radians(angle) / 0.005)  # 0.01
        lift_pos = np.array(lift_pos)
        _, pos = solveIK(lift_pos.reshape(3, 6))
        init_point = np.array(pos)
        base_traj = []

        for step in range(timestep):
            traj_ = init_point + np.array([0, 0, 0, 0, 0, 0,
                                           0, (step * 0.0025), (step * 0.0025), 0, (step * -0.0025), (step * -0.0025),
                                           0, (step * -0.005), (step * -0.0025), 0, (step * 0.005), (step * 0.0025)])
            base_traj.append(traj_)
            self.env.step(traj_)

        return base_traj[-1]

    # 夹物体
    def grasp_object(self, adjust_pos):
        adjust_pos = np.array(adjust_pos)
        init_pos = solveFK(adjust_pos).flatten()
        middle_pos_4 = np.hstack((0.7, 0.7, init_pos[2:6], 0.3, -0.3, init_pos[8:12], 0, 0, init_pos[14:18]))
        end_pos_4 = np.hstack((0.83, 0.83, init_pos[2:6], 0.1, -0.1, init_pos[8:12], 0.1, 0.1, init_pos[14:18]))
        waypoints_12 = np.vstack((init_pos, middle_pos_4, end_pos_4))
        time_vector_2 = [0, 1, 2]
        self.move_leg(waypoints_12, time_vector_2, total_time=2)

        return end_pos_4

    # 在工作空间规划举起物体的路径
    def move_front_leg_workspace(self, timestep):
        self.is_moving = True
        self.timestep = timestep
        init_point = np.array([[0.85, 0.85, 0.23, 0.23, -0.4, -0.4],
                              [0.1, -0.1, 0.4, -0.4, 0.4, -0.4],
                              [0.1, 0.1, -0.15, -0.15, -0.15, -0.15]])
        traj_1 = np.zeros((timestep, 3))
        traj_2 = np.zeros((timestep, 3))
        for step in range(timestep):
            traj_1[step] = self.trajplanner.front_leg_workspace_traj(step, leg_index=0)
            traj_2[step] = self.trajplanner.front_leg_workspace_traj(step, leg_index=1)
            traj_ = np.hstack((traj_1[step].reshape(-1,1),traj_2[step].reshape(-1,1),init_point[:,[2,3,4,5]]))
            self.env.step(traj_,sleep=0.02)
        return True

    # 在关节空间规划举起物体的路径
    def lift_leg_jointspace(self, base_pos, timestep):
        self.is_moving = True
        self.timestep = timestep
        lift_pos= np.array(base_pos)
        init_jointspace, pos = solveIK(lift_pos.reshape(3, 6))
        init_point = np.array(pos)
        # init_point = base_pos
        traj_1 = np.zeros((timestep, 3))
        joint1_forces = []
        joint1_torques = []
        joint2_forces = []
        joint2_torques = []
        joint3_forces = []
        joint3_torques = []
        trajs = []

        for step in range(timestep):
            # traj_1[step] = self.grasper.cubic_interpolation_traj(init_point, step)  # 三次多项式插值
            # traj_1[step] = self.grasper.polynomial_interpolation_path(init_point, step)  # 五次多项式插值
            traj_1[step] = self.grasper.front_leg_jointspace_traj(init_point, step, leg_index=0)  # 能量函数
            # traj_dt = traj_1[step] - traj_1[step-1]
            # traj_dt= (trajectory(t + dt)[0] - x) / dt, (trajectory(t + dt)[1] - y) / dt, (trajectory(t + dt)[2] - z) / dt
            traj_ = np.hstack((traj_1[step], (-1) * traj_1[step], init_point[6:18]))
            trajs.append(traj_)
            position, velocity, torque = self.env.step(traj_)
            joint1_torque = torque[0]
            joint2_torque = torque[1]
            joint3_torque = torque[2]

            joint1_force = np.linalg.norm(joint1_torque[0:3])
            joint1_torque = np.linalg.norm(joint1_torque[3:6])
            joint2_force = np.linalg.norm(joint2_torque[0:3])
            joint2_torque = np.linalg.norm(joint2_torque[3:6])
            joint3_force = np.linalg.norm(joint3_torque[0:3])
            joint3_torque = np.linalg.norm(joint3_torque[3:6])

            power = abs(joint1_torque) * abs(velocity[0]) + abs(joint2_torque) * abs(velocity[1]) + abs(joint3_force) * abs(velocity[2])
            # energy, _ = quad(power, start_time=0, end_time=1)

            joint2_forces.append(joint2_force)
            joint2_torques.append(joint2_torque)
            # wandb.log({
            #     "F1": joint1_force,
            #     "F2": joint2_force,
            #     "F3": joint3_force,
            #     "T1": joint1_torque,
            #     "T2": joint2_torque,
            #     "T3": joint3_torque,
            #     "Energy2": power
            # })
        joint2_forces = np.array(joint2_forces)
        joint2_torques = np.array(joint2_torques)
        # wandb.finish()
        plt.plot(range(timestep), joint2_forces, label='Force')
        plt.plot(range(timestep), joint2_torques, label='Torque')
        plt.xlabel('Simulation Step')
        plt.ylabel('Joint Torque(Nm)')
        plt.legend(loc='upper right')  # Display the legend
        plt.title('Joint Torque over Simulation Steps')
        plt.show()
        print(trajs[-1])
        return trajs[-1]

    # 放下腿
    def release_leg_jointspace(self, liftup_pos, timestep):
        self.is_moving = True
        self.timestep = timestep
        init_point = np.array(liftup_pos)
        init_point.reshape(1, 18)
        traj_1 = np.zeros((timestep, 3))
        trajs = []
        for step in range(timestep):
            traj_1[step] = self.grasper.release_jointspace_traj(init_point, step, leg_index=0)
            traj_ = np.hstack((traj_1[step], (-1) * traj_1[step], init_point[6:18]))
            trajs.append(traj_)
            self.env.step(traj_)
        return trajs[-1]

    # 支撑腿复位
    def reset_leg(self, init_pos):
        leg_clearance = -0.05
        time_vector_1 = [0, 0.5, 1]

        # leg 6
        middle_pos_3 = np.hstack((init_pos[0:5], (init_pos[5] + init_leg[5]) / 2,
                                  init_pos[6:11], (init_pos[11] + init_leg[11]) / 2,
                                  init_pos[12:17], leg_clearance))
        end_pos_3 = np.hstack((init_pos[0:5], init_leg[5], init_pos[6:11], init_leg[11], init_pos[12:18]))
        waypoints_6 = np.vstack((init_pos, middle_pos_3, end_pos_3))
        self.move_leg(waypoints_6, time_vector_1, total_time=1)

        # leg 5
        init_pos = end_pos_3
        middle_pos_2 = np.hstack((init_pos[0:4], (init_pos[4] + init_leg[4]) / 2,
                                  init_pos[5:10], (init_pos[10] + init_leg[10]) / 2,
                                  init_pos[11:16], leg_clearance, init_pos[17]))
        end_pos_2 = np.hstack((init_pos[0:4], init_leg[4], init_pos[5:10], init_leg[10], init_pos[11:18]))
        waypoints_5 = np.vstack((init_pos, middle_pos_2, end_pos_2))
        self.move_leg(waypoints_5, time_vector_1, total_time=1)

        # leg 3, 4
        init_pos = end_pos_2
        middle_pos = np.hstack((init_pos[0:2], (init_pos[2] + init_leg[2]) / 2, (init_pos[3] + init_leg[3]) / 2,
                                init_pos[4:8], (init_pos[8] + init_leg[8]) / 2, (init_pos[9] + init_leg[9]) / 2,
                                init_pos[10:14], leg_clearance, leg_clearance, init_pos[16:18]))
        end_pos = init_leg
        waypoints_34 = np.vstack((init_pos, middle_pos, end_pos))
        self.move_leg(waypoints_34, time_vector_1, total_time=1)


if __name__ == '__main__':
    from Hebi_test import Hebi
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import functions as f

    hebi = Hebi(real_robot_control=1, pybullet_on=1)
    hebi.env.camerafollow = 1

    # 调整支撑腿位置
    leg_4_pos = [0.23, 0.4]  # leg 3
    leg_5_pos = [-0.4, 0.4]  # leg 6
    stand_pos = hebi.adjust_leg(leg_4_pos, leg_5_pos)

    # 调整base角度
    base_pos = hebi.adjust_base(stand_pos, 15)

    # 举起前腿取物体
    lift_pos = hebi.grasp_object(base_pos)

    input("按回车键继续...")

    # 举到背上
    liftup_pos = hebi.lift_leg_jointspace(lift_pos, timestep=730)

    input("按回车键继续...")

    # 腿放下
    end_joint = hebi.release_leg_jointspace(liftup_pos, timestep=150)

    end_pose = f.solveFK(end_joint)
    waypoints = np.vstack((end_pose.reshape(1, 18), stand_pos))
    time_vector = [0, 1]
    hebi.move_leg(waypoints, time_vector, total_time=1)

    # 腿复位
    hebi.reset_leg(stand_pos)

    time.sleep(2)

    hebi.stop()
    hebi.disconnect()
