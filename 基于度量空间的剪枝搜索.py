import numpy as np
import time


def euclidean_dist(p1, p2):
    """计算欧几里得距离：对应度量空间中的 d(x, y)"""
    return np.linalg.norm(p1 - p2)


# ==========================================
# 1. 初始化数据集 (模拟矿点或资源点)
# ==========================================
np.random.seed(42)#固定种子，使产生的随机数序列完全一样，像工程里的权重通常需要固定，不然跑出来的结果次次不一样
N = 1000  # 总点数
dim = 5  # 维度 (5维空间)
data = np.random.rand(N, dim)

# ==========================================
# 2. 选取参考点 (Pivots)
# 法则：尽量选在空间的“边界”或“角落”，这样投影效果最好
# ==========================================
pivots_idx = [0]  # 先随机选第一个点 （!最远优先遍历）
for _ in range(4):  # 我们一共选 5 个参考点
    # 找到距离当前已选参考点集最远的点
    #这个计算量太大：
    # dists = [min([euclidean_dist(data[i], data[p]) for p in pivots_idx]) for i in range(N)]
    # 这一行替代你那行复杂的列表推导式
    # data[pivots_idx] 是一个 (k, 5) 的矩阵
    # data[:, np.newaxis] 是一个 (N, 1, 5) 的矩阵
    # 这样一次性算出所有点对所有参考点的距离
    dists_matrix = np.linalg.norm(data[:, np.newaxis] - data[pivots_idx], axis=2)
    dists = np.min(dists_matrix, axis=1)
    pivots_idx.append(np.argmax(dists))

pivots = data[pivots_idx]
print(f"成功选取了 {len(pivots)} 个参考点作为空间的'骨架'。")

# ==========================================
# 3. 预计算 (离线阶段)
# 计算数据库中所有点到这 5 个参考点的距离
# ==========================================
# 每一行代表一个点，每一列代表到某个参考点的距离
pivot_distances = np.array([[euclidean_dist(p, a) for a in pivots] for p in data])

# ==========================================
# 4. 在线查询阶段
# 任务：寻找距离 query_point 半径 0.5 以内的所有点
# ==========================================
query_point = np.random.rand(dim)
radius = 0.5

print(f"\n查询任务：在 {N} 个点中寻找半径 {radius} 内的近邻...")

# --- 方法 A: 暴力搜索 (不讲数学武德) ---
start_bf = time.time()
bf_matches = []
for i in range(N):
    if euclidean_dist(query_point, data[i]) <= radius:
        bf_matches.append(i)
bf_time = time.time() - start_bf

# --- 方法 B: 拓扑剪枝 (利用反向三角不等式) ---
start_pruning = time.time()
pruned_matches = []
dist_calcs = 0  # 记录真正昂贵的距离计算次数
skipped = 0

# 第一步：只计算查询点到 5 个参考点的距离 (代价极低)
q_to_pivots = [euclidean_dist(query_point, a) for a in pivots]

# 第二步：利用逻辑排除点
for i in range(N):
    can_prune = False
    for j in range(len(pivots)):
        # 【数学核心】：|d(q, a) - d(p, a)| <= d(q, p)
        # 如果左边这一项已经大于半径了，那么真实的 d(q, p) 绝对不可能小于等于半径！
        if abs(q_to_pivots[j] - pivot_distances[i, j]) > radius:
            can_prune = True
            break  # 只要有一个参考点说它太远，直接 pass

    if not can_prune:
        # 只有无法被排除的点，才需要进行真正的、昂贵的距离计算
        dist_calcs += 1
        if euclidean_dist(query_point, data[i]) <= radius:
            pruned_matches.append(i)
    else:
        skipped += 1

pruning_time = time.time() - start_pruning

# ==========================================
# 5. 结果对比
# ==========================================
print("-" * 30)
print(f"暴力搜索计算次数: {N} 次")
print(f"拓扑剪枝计算次数: {dist_calcs + len(pivots)} 次")
print(f"成功跳过了 {skipped} 个点的计算 (节省了 {skipped / N * 100:.2f}% 的工作量)")
print(f"结果验证: {'一致' if set(bf_matches) == set(pruned_matches) else '错误'}")
print("-" * 30)