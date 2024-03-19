import pybullet as p
import pybullet_data
import numpy as np
import os, sys
import time
import robot_setup
from robot_setup.hebiKinematics import *
from functions import hebi2bullet, solveIK
import matplotlib.pyplot as plt

class HebiEnv:
    def __init__(self, real_robot_control=True, pybullet_on=True, visualiser=True, camerafollow=True):
        self.real_robot_control = real_robot_control
        self.visualiser = visualiser
        self.camerafollow = camerafollow
        self.dt = 1 /240
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.robot_connect()
        self.error = np.zeros((18,))
        self.pybullet_on = pybullet_on
        if self.pybullet_on:
            self._load_env()
        self._init_robot()

    def step(self, targetPositions, iteration=1, sleep='auto'):
        '''
        Advance the simulation and physical robot by one step
        :param targetPositions: the target position of the robot, either in the form of workspace command (shape=(3,6)) or jointspace command (shape=(18,))
        :param iteration: the number of iterations to reach the target position, default is 1. If the given target position is not possible to reach within 1 step, more than 1 iteration could be set
        :param sleep: the time to sleep after each step, default is 'auto', which means the time to sleep is automatically calculated based on the given frequency
        return: None
        '''
        # judge the input target position is workspace command or jointspace command
        if np.shape(targetPositions) == (3,6): # workspace command
            jointspace_command2bullet, jointspace_command2hebi = solveIK(targetPositions)
        elif np.shape(targetPositions) == (18,): # jointspace command
            jointspace_command2bullet, jointspace_command2hebi = hebi2bullet(targetPositions), targetPositions
        else:
            raise ValueError('Command that Yuna cannot recognise, please input either workspace command whose shape(targetPositions)=(3,6), or jointspace command whose shape(targetPositions)=(18,)')
        
        for i in range(iteration): # iteration is usually 1, but if the given target position is not possible to reach within 1 step, more than 1 iteration could be set
            t_start = time.perf_counter()
            # real robot control
            if self.real_robot_control:
                self.group_command.position = jointspace_command2hebi - 0.3 * self.error
                self.hexapod.send_command(self.group_command)
                while True:
                    try:
                        self.group_feedback = self.hexapod.get_next_feedback(reuse_fbk=self.group_feedback)
                        self.error = self.group_feedback.position - jointspace_command2hebi
                        break
                    except:
                        pass
            p.enableJointForceTorqueSensor(self.HebiID, self.actuator[0], enableSensor=True)
            p.enableJointForceTorqueSensor(self.HebiID, self.actuator[1], enableSensor=True)
            p.enableJointForceTorqueSensor(self.HebiID, self.actuator[2], enableSensor=True)
            p.enableJointForceTorqueSensor(self.HebiID, self.actuator[3], enableSensor=True)

            # pybullet control
            if self.pybullet_on:
                p.setJointMotorControlArray(
                    bodyIndex=self.HebiID,
                    jointIndices=self.actuator, 
                    controlMode=p.POSITION_CONTROL, 
                    targetPositions=jointspace_command2bullet)
                joint_states = p.getJointStates(self.HebiID, range(p.getNumJoints(self.HebiID)))
                joint_velocity = [state[1] for state in joint_states]
                joint_torques = [state[2] for state in joint_states]
                p.stepSimulation()
                if self.camerafollow:
                    self._cam_follow()
            t_stop = time.perf_counter()
            t_step = t_stop - t_start
            if sleep == 'auto':
                time.sleep(max(0, self.dt - t_step))
            else:
                time.sleep(sleep)
        return joint_velocity, joint_torques

    def close(self):
        '''
        Close the pybullet simulation, disable the real robot motors, and terminate the program
        :return: None
        '''
        if self.real_robot_control:
                arr = np.zeros([1, 18])[0]
                self.group_command.effort = arr
                self.group_command.position = np.nan * arr # np.nan-not a number
                self.group_command.velocity_limit_max = arr
                self.group_command.velocity_limit_min = arr
                self.hexapod.send_command(self.group_command)
        if self.pybullet_on:
            try:
                p.disconnect()
            except p.error as e:
                print('Termination of simulation failed:', e)
        sys.exit()

    def robot_connect(self):
        '''
        Initialise connection to the real robot
        :return: xmk, imu, hexapod, fbk_imu, fbk_hp, group_command, group_feedback
        '''
        if self.real_robot_control:
            xmk, imu, hexapod, fbk_imu, fbk_hp = robot_setup.setup_xmonster()
            group_command = hebi.GroupCommand(hexapod.size)
            group_feedback = hebi.GroupFeedback(hexapod.size)
            hexapod.feedback_frequency = 100.0
            hexapod.command_lifetime = 0
            while True:
                group_feedback = hexapod.get_next_feedback(reuse_fbk=group_feedback)
                if type(group_feedback) != None:
                    break
            return xmk, imu, hexapod, fbk_imu, fbk_hp, group_command, group_feedback
        else:
            return HexapodKinematics(), False, False, False, False, False, False

    def _load_env(self):
        '''
        Load and initialise the pybullet simulation environment
        :return: None
        '''
        # initialise interface
        if self.visualiser:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        # physical parameters
        self.gravity = -9.81
        self.friction = 0.7
        # load ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.groundID = p.loadURDF('plane.urdf')
        p.setGravity(0, 0, self.gravity)
        p.changeDynamics(self.groundID, -1, lateralFriction=self.friction)
        # load Yuna robot
        Hebi_init_pos = [0,0,0.5]
        Hebi_init_orn = p.getQuaternionFromEuler([0,0,0])
        Hebi_file_path = os.path.abspath(os.path.dirname(__file__)) + '/urdf/hebi.urdf'
        # self.foundationID = p.loadURDF('/urdf/foundation.urdf', np.array([0.7, 0, 0]))
        # self.boxID = p.loadURDF('/urdf/box.urdf', np.array([0.7, 0, 0]))
        self.HebiID = p.loadURDF(Hebi_file_path, Hebi_init_pos, Hebi_init_orn)
        self.joint_num = p.getNumJoints(self.HebiID)
        self.actuator = [i for i in range(self.joint_num) if p.getJointInfo(self.HebiID,i)[2] != p.JOINT_FIXED]

        if self.visualiser:
            # self._add_reference_line()
            self._cam_follow()
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            
    def _init_robot(self):
        '''
        Initialise the robot to the neutral position in the pybullet simulation
        :return: None
        '''
        # parameters
        self.h = 0.12 #0.2249 # body height
        self.eePos = np.array( [[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                                [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                                [-self.h,   -self.h,    -self.h,    -self.h,    -self.h,    -self.h]]) # neutral position for the robot
        init_pos = self.eePos.copy()
        self.step(init_pos, iteration=65, sleep='auto')
  
    def _cam_follow(self):
        '''
        Follow the robot with the camera in the pybullet simulation
        :return: None
        '''
        def _get_body_pose(self):
            '''
            Get the position and orientation of the robot in the pybullet simulation
            :return pos: position of the robot in world frame
            :return orn: orientation of the robot in world frame in Euler angles
            '''
            pos, orn = p.getBasePositionAndOrientation(self.HebiID)
            return pos, p.getEulerFromQuaternion(orn)
        cam_pos, cam_orn = _get_body_pose(self)
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-225, cameraPitch=-35, cameraTargetPosition=cam_pos)

    def _add_reference_line(self):
        '''
        Add some reference lines to the pybullet simulation
        :return: None
        '''
        p.addUserDebugLine(lineFromXYZ=[-100,-100,0], lineToXYZ=[100,100,0], lineColorRGB=[0.5,0,0], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-100,100,0], lineToXYZ=[100,-100,0], lineColorRGB=[0.5,0,0], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-100,-173.2,0], lineToXYZ=[100,173.2,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-100,173.2,0], lineToXYZ=[100,-173.2,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-173.2,-100,0], lineToXYZ=[173.2,100,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-173.2,100,0], lineToXYZ=[173.2,-100,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-100,0,0], lineToXYZ=[100,0,0], lineColorRGB=[0,0,0], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[0,-100,0], lineToXYZ=[0,100,0], lineColorRGB=[0,0,0], lineWidth=1)

    # def getjointforce(self):
    #     '''
    #     Get the joint force of the robot
    #     :return: joint force of the robot
    #     '''
    #     # joint_states = p.getJointStates(self.HebiID, range(p.getNumJoints(self.HebiID)))
    #     joint_forces = [state[3] for state in joint_states]
    #     joint_torques = [state[2] for state in joint_states]
    #     return joint_torques[1]


if __name__=='__main__':
    # test code
    hebienv = HebiEnv(real_robot_control=0)
    time.sleep(60)
    hebienv.close()
