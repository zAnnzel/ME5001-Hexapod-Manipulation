import wandb
import numpy as np

# 初始化Wandb
wandb.init(project='Hebi-robot', name='cubic', config={})

# 读取feedback_velocity.txt
# with open("./data-15-三次/feedback_velocity.txt", "r") as file:
#     content = file.read()
#     parts = content.split('[')
#
#     for part in parts:
#         if ']' in part:
#             match = part.split(']')[0]
#             numbers = [float(num) for num in match.split(' ') if num.strip()]
#             feedback_velocity.append(numbers)
#
# joint_1_velocity = [row[0] for row in feedback_velocity]
# joint_1_velocity = np.array(joint_1_velocity)
# joint_2_velocity = [row[0] for row in feedback_velocity]
# joint_2_velocity = np.array(joint_2_velocity).reshape(-1, 1)
#
# for i, value in enumerate(joint_1_velocity):
#     # 使用 wandb.log 函数记录数据点
#     wandb.log({"joint_1_velocity": value}, step=i)


# 读取数据
def plot_data(file_name, data_type, joint_index):
    arrays = {}
    array_name = f"{data_type}_{joint_index}"
    datas = []
    with open("./data-15-五次/" + file_name, "r") as file:
        content = file.read()
        parts = content.split('[')

        for part in parts:
            if ']' in part:
                match = part.split(']')[0]
                numbers = [float(num) for num in match.split(' ') if num.strip()]
                datas.append(numbers)

    lines = [row[joint_index] for row in datas]
    arrays[array_name] = np.array(lines)
    for i, value in enumerate(arrays[array_name]):
        # 使用 wandb.log 函数记录数据点
        wandb.log({f"{data_type}_{joint_index}": value}, step=i)


# plot_data("feedback_effort.txt", "feedback_effort", 0)
plot_data("feedback_velocity.txt", "feedback_velocity", 0)

# 绘制曲线
# wandb.log({
#     # "command": jointspace_command2hebi,
#     # "Position": feedback_position[0:6],
#     "V1": joint_1_velocity,
#     "V2": joint_2_velocity,
#     # "Effort": feedback_effort
# })

# 关闭Wandb
wandb.finish()


# # 读取data.txt
# with open("data.txt", "r") as file:
#     lines = file.readlines()
#
#     for i, line in enumerate(lines):
#         line = line.replace('[', '').replace(']', '')
#         remainder = i % 16
#         if remainder in [0, 1, 2, 3, 4]:
#             jointspace_command2hebi.append(list(map(float, line.split())))
#         elif remainder in [5, 6, 7]:
#             feedback_position.append(list(map(float, line.split())))
#         elif remainder in [8, 9, 10]:
#             feedback_velocity.append(list(map(float, line.split())))
#         # elif remainder == 8:
#         #     feedback_velocity_1.append(list(map(float, line.split())))
#         # elif remainder == 9:
#         #     feedback_velocity_2.append(list(map(float, line.split())))
#         # elif remainder == 10:
#         #     feedback_velocity_3.append(list(map(float, line.split())))
#         elif remainder in [11, 12, 13, 14, 15]:
#             feedback_effort.append(list(map(float, line.split())))
#
#     # feedback_velocity = np.hstack((feedback_velocity_1, feedback_velocity_2, feedback_velocity_3))
#     # 打印结果或进一步处理数组
# print("feedback_velocity:", feedback_velocity)


# # 读取feedback_effort.txt
# with open("./data-15-三次/feedback_effort.txt", "r") as file:
#     content = file.read()
#     parts = content.split('[')
#
#     for part in parts:
#         if ']' in part:
#             match = part.split(']')[0]
#             numbers = [float(num) for num in match.split(' ') if num.strip()]
#             feedback_effort.append(numbers)
#
# joint_1_effort = [row[1] for row in feedback_effort]
# joint_1_effort = np.array(joint_1_effort)
#
# for i, value in enumerate(joint_1_velocity):
#     # 使用 wandb.log 函数记录数据点
#     wandb.log({"joint_1_effort": value}, step=i)

def read_data(file_name, data_type, joint_index):
    arrays = {}
    array_name = f"{data_type}_{joint_index}"
    datas = []
    with open("./data-15-五次/" + file_name, "r") as file:
        content = file.read()
        parts = content.split('[')

        for part in parts:
            if ']' in part:
                match = part.split(']')[0]
                numbers = [float(num) for num in match.split(' ') if num.strip()]
                datas.append(numbers)

    lines = [row[joint_index] for row in datas]
    arrays[array_name] = np.array(lines)
    for i, value in enumerate(arrays[array_name]):
        # 使用 wandb.log 函数记录数据点
        wandb.log({f"{data_type}_{joint_index}": value}, step=i)