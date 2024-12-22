import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

# 定义目标函数（示例）
def objective_function(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))

# 采集函数：期望改进（EI）
def expected_improvement(x, model, y_min):
    x = np.array(x).reshape(-1, 1)
    mu, sigma = model.predict(x, return_std=True)
    sigma = sigma.reshape(-1, 1)

    with np.errstate(divide='warn'):
        Z = (mu - y_min) / sigma
        ei = (mu - y_min) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return -ei  # 最优化时取负号

# TPE目标分布模型
def tpe_model(samples, scores):
    sorted_idx = np.argsort(scores)
    l_idx = sorted_idx[:len(sorted_idx) // 2]
    g_idx = sorted_idx[len(sorted_idx) // 2:]

    l_samples = samples[l_idx]
    g_samples = samples[g_idx]

    def l_pdf(x):
        return np.sum(np.exp(-0.5 * ((x - l_samples) / 0.1) ** 2))

    def g_pdf(x):
        return np.sum(np.exp(-0.5 * ((x - g_samples) / 0.1) ** 2))

    return l_pdf, g_pdf

# HSBO 主函数
def hsbo_tpe(objective, bounds, n_iters=50, init_samples=5):
    # 初始样本
    X = np.random.uniform(bounds[0], bounds[1], size=(init_samples, 1))
    y = np.array([objective(x[0]) for x in X])

    # 高斯过程模型
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

    for i in range(n_iters):
        # 用当前数据拟合高斯过程模型
        gp.fit(X, y)

        # 当前最优值
        y_min = np.min(y)

        # 使用采集函数（EI）优化
        def min_obj(x):
            return expected_improvement(x, gp, y_min)

        res = minimize(min_obj, x0=np.random.uniform(bounds[0], bounds[1]), bounds=[bounds])
        x_next = res.x

        # TPE建模
        l_pdf, g_pdf = tpe_model(X.flatten(), y)
        tpe_score = l_pdf(x_next) / g_pdf(x_next)

        if tpe_score < 1.0:  # 使用TPE的逻辑
            x_next = np.random.uniform(bounds[0], bounds[1])

        # 评估新样本
        y_next = objective(x_next[0])

        # 更新数据
        X = np.vstack((X, x_next))
        y = np.append(y, y_next)

        print(f"Iteration {i+1}/{n_iters}: Best y = {y_min}")

    return X, y

# 参数范围和运行
bounds = (0, 2)
X, y = hsbo_tpe(objective_function, bounds)

# 找到的最优点
best_idx = np.argmin(y)
print("Best x:", X[best_idx])
print("Best y:", y[best_idx])
