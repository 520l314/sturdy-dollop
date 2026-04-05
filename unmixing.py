import math
import time
import numpy as np
import torch
from torch.optim import Adam

# ----------------------------------------------------------------------
# 全局设备配置
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# 辅助函数（与专利论文一致）
# ----------------------------------------------------------------------
def solve_min_norm_2_loss(grad_1, grad_2):
    """
    计算 Pareto 下降方向 φ*(x)   (论文公式 15)
    """
    v1v1 = torch.sum(grad_1 * grad_1, dim=0)
    v2v2 = torch.sum(grad_2 * grad_2, dim=0)
    v1v2 = torch.sum(grad_1 * grad_2, dim=0)
    denominator = v1v1 + v2v2 - 2 * v1v2
    denominator[denominator == 0] = 1e-8
    gamma = -1.0 * ((v1v2 - v2v2) / denominator)
    gamma[v1v2 >= v1v1] = 0.999
    gamma[v1v2 >= v2v2] = 0.001
    gamma = torch.clamp(gamma, 0.001, 0.999)
    gamma = gamma.view(1, -1)
    phi_star = gamma * grad_1 + (1. - gamma) * grad_2
    return phi_star

def median_heuristic(tensor):
    """用于核函数带宽的中位数启发式"""
    tensor = tensor.detach().flatten()
    if tensor.numel() == 0:
        return torch.tensor(1.0, device=tensor.device)
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.0

def kernel_functional_rbf(x):
    """
    计算 RBF 核矩阵，用于排斥项 (论文公式 16)
    x: (n_endmembers, pop_size)
    """
    n = x.shape[1]
    dist_matrix = torch.cdist(x.t(), x.t(), p=2)
    pairwise_distance_sq = dist_matrix.pow(2)
    h = median_heuristic(pairwise_distance_sq) / (math.log(n) + 1e-8)
    kernel_matrix = torch.exp(-pairwise_distance_sq / (h + 1e-8))
    return kernel_matrix

def compute_pareto_descent_direction(x, grad_1, grad_2, alpha=0.5):
    """
    计算最终下降方向 φ(x_i) (论文公式 17)
    """
    n = x.shape[1]
    phi_star = solve_min_norm_2_loss(grad_1, grad_2)          # φ*(x)
    # 计算核函数梯度项
    kernel = kernel_functional_rbf(x)
    kernel_grad_list = []
    for i in range(n):
        kg = torch.autograd.grad(kernel[i, :].sum(), x, retain_graph=True)[0]
        if kg is None:
            kg = torch.zeros_like(x)
        kernel_grad_list.append(kg[:, i:i+1])
    kernel_grad = torch.cat(kernel_grad_list, dim=1)
    # 最终方向
    phi = (1.0 / n) * phi_star - alpha * kernel_grad
    return phi

def binary_hash(data, beta=0.9):
    """贪心哈希二值化 (论文公式 18)"""
    out = data.copy()
    out[data < beta] = 0
    out[data >= beta] = 1
    return out

def loss_function(x, library, observation, abundance_est, target_k):
    """
    双目标函数 (论文公式 12)
    f1: 重构误差
    f2: 稀疏误差 |‖s‖₁ - k|
    """
    pop_size = x.shape[1]
    loss_f1 = []
    loss_f2 = []
    for i in range(pop_size):
        s_i = x[:, i]
        weighted_abundance = abundance_est * s_i.unsqueeze(1)
        reconstruction = torch.mm(library, weighted_abundance)
        f1 = torch.norm(observation - reconstruction, p='fro')
        loss_f1.append(f1)
        l1_norm = torch.sum(s_i)
        f2 = torch.abs(l1_norm - target_k)
        loss_f2.append(f2)
    return torch.stack(loss_f1), torch.stack(loss_f2)

# ----------------------------------------------------------------------
# 主算法：GMOGH 高光谱解混（封装为平台调用接口）
# ----------------------------------------------------------------------
def gmogh_unmixing(Y, library, target_k, beta=0.9,
                   max_iters=3000, lr=0.05, pop_size=30,
                   init_scale=1.0, alpha=0.5, verbose=False):
    """
    专利 GMOGH 高光谱解混算法（完整实现）

    参数:
        Y: 高光谱数据，形状 (n_pixels, n_bands)
        library: 端元库，形状 (n_bands, n_endmembers)
        target_k: 目标端元数量
        beta: 二值化阈值 (0~1)
        max_iters: 最大迭代次数
        lr: 学习率
        pop_size: 种群大小
        init_scale: 初始解缩放因子
        alpha: 核排斥项权重
        verbose: 是否打印进度

    返回:
        selected_idx: 选中的端元在库中的索引 (0-based), 形状 (target_k,)
        abundance: 丰度矩阵，形状 (target_k, n_pixels)
    """
    # 数据预处理（归一化）
    Y_np = Y.copy()
    lib_np = library.copy()
    if Y_np.max() > 0:
        Y_np = Y_np / Y_np.max()
    if lib_np.max() > 0:
        lib_np = lib_np / lib_np.max()

    Y_tensor = torch.from_numpy(Y_np.T).float().to(DEVICE)          # (bands, pixels)
    library_tensor = torch.from_numpy(lib_np).float().to(DEVICE)    # (bands, M)

    # 预计算丰度估计（伪逆 + ReLU）
    pinv_lib = torch.pinverse(library_tensor)
    abundance_est = torch.relu(torch.mm(pinv_lib, Y_tensor))        # (M, pixels)

    n_endmembers = library.shape[1]

    # 初始化种群
    x = torch.rand((n_endmembers, pop_size), device=DEVICE) * init_scale
    x.requires_grad = True

    best_solution = None
    best_loss = float('inf')

    if verbose:
        print(f"   GMOGH: k={target_k}, pop={pop_size}, lr={lr}, iters={max_iters}")

    for it in range(max_iters):
        loss1, loss2 = loss_function(x, library_tensor, Y_tensor, abundance_est, target_k)
        total_loss = loss1 + loss2

        # 更新最佳解
        min_idx = torch.argmin(total_loss).item()
        if total_loss[min_idx] < best_loss:
            best_loss = total_loss[min_idx].item()
            best_solution = x[:, min_idx].detach().clone()

        # 计算两个目标的梯度
        # 目标1
        x.grad = None
        loss1.sum().backward(retain_graph=True)
        grad1 = x.grad.detach().clone()
        # 目标2
        x.grad = None
        loss2.sum().backward()
        grad2 = x.grad.detach().clone()

        # 梯度归一化
        grad1 = torch.nn.functional.normalize(grad1, dim=0)
        grad2 = torch.nn.functional.normalize(grad2, dim=0)

        # 计算 Pareto 下降方向
        phi = compute_pareto_descent_direction(x, grad1, grad2, alpha=alpha)

        # 梯度上升/下降？论文中使用的是梯度上升（最大化核排斥），但这里 phi 已经考虑了符号
        # 实际更新： x = x + epsilon * phi
        with torch.no_grad():
            x.data += lr * phi
            x.data.clamp_(0.0, 1.0)

        # 每隔一定迭代，使用二值化解指导连续解（贪心哈希）
        if it % 10 == 0:
            with torch.no_grad():
                s_bin = torch.from_numpy(binary_hash(x.cpu().numpy(), beta=beta)).float().to(DEVICE)
                # 软更新：混合当前连续解与二值化解
                x.data = 0.9 * x.data + 0.1 * s_bin

        if verbose and (it+1) % 500 == 0:
            print(f"      Iter {it+1}/{max_iters}, Best loss = {best_loss:.4f}")

    # 提取最佳二值化解
    if best_solution is None:
        best_solution = x[:, torch.argmin(loss1+loss2)].detach().clone()
    final_s = binary_hash(best_solution.cpu().numpy(), beta=beta).flatten()
    selected_idx = np.where(final_s == 1)[0]
    if len(selected_idx) != target_k:
        # 如果数量不对，强制取 top-k
        selected_idx = np.argsort(best_solution.cpu().numpy())[::-1][:target_k]

    selected_idx = np.sort(selected_idx)

    # 最终丰度反演（使用原始未归一化的数据，以保证物理意义）
    lib_sub = library[:, selected_idx]          # (bands, K)
    # 伪逆求解丰度
    abundance = np.linalg.pinv(lib_sub) @ Y.T   # (K, pixels)
    abundance = np.maximum(abundance, 0)        # 非负
    # 和为1归一化
    sum_ab = np.sum(abundance, axis=0, keepdims=True)
    sum_ab[sum_ab == 0] = 1.0
    abundance = abundance / sum_ab

    return selected_idx, abundance               # abundance shape (K, n_pixels)