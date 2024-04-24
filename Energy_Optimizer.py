import numpy as np
import pyswarms as ps

def trajectory_energy(x, start_position, end_position):
    """ 计算机械臂从起点到终点的能量消耗。

    参数:
    x -- np.array, 粒子表示中间过渡状态的关节角度，形状为(n_particles, n_dimensions-2)
    start_position -- 起点关节角度
    end_position -- 终点关节角度

    返回:
    np.array -- 每个粒子的能量消耗
    """
    # 将起点和终点加入到粒子的位置表示中
    full_trajectory = np.column_stack([start_position, x, end_position])

    # 简化能量计算示例，实际应用中应该更复杂
    energy = np.sum(full_trajectory ** 2, axis=1)
    return energy


def constrained_energy_consumption(x):
    """计算考虑起点和终点约束的能量消耗。

    参数:
    x -- np.array, 关节角度数组，形状为(n_particles, n_dimensions)

    返回:
    np.array -- 每个粒子的总能量消耗，包括惩罚
    """
    # 基本能量消耗模型，同前面定义
    basic_energy = np.sum(x ** 2, axis=1)

    # 添加起点和终点的惩罚
    start_penalty = np.linalg.norm(x - start_pos, axis=1) ** 2
    end_penalty = np.linalg.norm(x - end_pos, axis=1) ** 2

    # 总能量消耗
    total_energy = basic_energy + 100 * start_penalty + 100 * end_penalty
    return total_energy


# 定义起点和终点关节角度
start_pos = np.array([-0.49188775, 0.15800083,  0.01668477])  # 起点角度
end_pos = np.array([-0.53168775, -2.82699917, -2.37131523])  # 终点角度

# 设定粒子群优化器的参数
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 创建一个粒子群优化器
# - n_particles: 粒子数
# - dimensions: 参数维数，此例中为3（三轴机械臂）
# - options: 优化参数
# 假设我们只优化起点和终点之间的一个中间点（三轴机械臂）
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=3, options=options)
#
# # 进行优化
# # - iters: 迭代次数
# cost, pos = optimizer.optimize(trajectory_energy, iters=100, start_position=start_pos, end_position=end_pos)

# 使用新的目标函数执行PSO优化
cost, pos = optimizer.optimize(constrained_energy_consumption, iters=100)


print("最优关节角度:", pos)
print("最小能量消耗:", cost)