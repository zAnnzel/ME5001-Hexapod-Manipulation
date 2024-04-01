import wandb

# 初始化Wandb
wandb.init(project='Hebi-robot', name='Pos-Vel-Effort', config={})

# 读取数据文件
with open("data.txt", "r") as file:
    lines = file.readlines()
    jointspace_command2hebi = []
    feedback_position = []
    feedback_velocity = []
    feedback_effort = []
    for line in lines:
        # 假设数据文件中每行包含两个数字，用空格分隔
        x, y = map(float, line.strip().split())
        jointspace_command2hebi.append(x)
        feedback_position.append(y)

# 绘制曲线
wandb.log({
    "command": jointspace_command2hebi,
    "Position": feedback_position[0:6],
    "Velocity": feedback_velocity[0:6],
    "Effort": feedback_effort
})

# 关闭Wandb
wandb.finish()
