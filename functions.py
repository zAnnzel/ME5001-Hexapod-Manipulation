import numpy as np
from robot_setup.hebiKinematics import HexapodKinematics

def rot(pos, angle, pivot=np.array([0,0,0])): #点围绕轴心点所在的z轴的旋转位置
    '''
    Calculate the rotated position of a point around z-axis where a pivot point lies
    :param pos: the position of the point to be rotated
    :param angle: the angle of rotation in radians, counterclockwise is positive
    :param pivot: the centre of rotation, only (x ,y) is effective
    :return: the rotated position
    '''
    c, s = np.cos(angle), np.sin(angle)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) #np.array()创建数组
    pos_ = np.matmul(rot_z, pos - pivot) #np.matmul()矩阵乘法
    return pos_ + pivot

def trans(pos, distance, angle): #点在某个方向上的平移位置
    '''
    Calculate the translated position of a point in a direction
    :param pos: the position of the point to be translated
    :param distance: the distance of translation
    :param angle: the direction of translation in radians, counterclockwise is positive
    :return: the translated position
    '''
    pos_ = np.copy(pos) #np.copy()复制数组
    pos_[0] += distance * np.cos(angle)
    pos_[1] += distance * np.sin(angle)
    return pos_

def hebi2bullet(jointspace_command2hebi):
    '''
    Convert the jointspace command for hebi to the jointspace command for pybullet
    :param jointspace_command2hebi: the jointspace command for hebi, whose size is (18,) and order is [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    :return: the jointspace command for pybullet, whose size is (18,) and order is [1,2,3,7,8,9,13,14,15,4,5,6,10,11,12,16,17,18]
    '''
    # reorder the jointspace command so that it can work in pybullet
    jointspace_command2bullet = jointspace_command2hebi[[0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17,]].copy()# reshaped the IK result: 123456->135246
    return jointspace_command2bullet
    
def solveIK(workspace_command): #求解逆运动学
    '''
    Use HexapodKinematics module to solve the inverse kinematics
    :param workspace_command: the workspace command, whose size is (3, 6)
    :return: the jointspace command for pybullet, the jointspace command for hebi
    '''
    xmk = HexapodKinematics()
    jointspace_command2hebi = xmk.getLegIK(workspace_command)
    jointspace_command2bullet = hebi2bullet(jointspace_command2hebi)
    return jointspace_command2bullet, jointspace_command2hebi

def solveFK(jointspace_command2hebi): #求解正向运动学
    '''
    Use HexapodKinematics module to solve the forward kinematics
    :param jointspace_command2hebi: the jointspace command for hebi, whose size is (18,) and order is [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    :return: the workspace command, whose size is (3, 6)
    '''
    xmk = HexapodKinematics()
    workspace_command = xmk.getLegPositions(np.array([jointspace_command2hebi]))
    return workspace_command

if __name__ == '__main__':
    workspace_command = np.array([[0.87, 0.87, 0.23, 0.23, -0.4, -0.4],
                                   [0.08, -0.08, 0.4, -0.4, 0.4, -0.4],
                                   [0, 0, -0.12, -0.12, -0.12, -0.12]])
    jointspace_command2bullet,jointspace_command2hebi = solveIK(workspace_command)
    print(jointspace_command2bullet)
    print(jointspace_command2hebi)