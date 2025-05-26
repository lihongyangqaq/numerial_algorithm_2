import math
import torch
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置显示参数
rcParams["font.family"] = "serif"

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# === 1. 生成高频函数数据 ===
N = 1000
x = np.linspace(0, 1, N)
y = x * np.sin(1000 * x)

# === 2. 减采样到200点（避免非正定）===
idx = np.linspace(0, N - 1, 200).astype(int)
train_x = torch.tensor(x[idx], dtype=torch.float32)
train_y = torch.tensor(y[idx], dtype=torch.float32)

# === 3. 构造GP模型 ===
class SpectralGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x.unsqueeze(-1), train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# === 4. 初始化模型和似然 ===
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
)
model = SpectralGPModel(train_x, train_y, likelihood)

# === 5. 训练模型 ===
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 150
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f"Iter {i:03d} - Loss: {loss.item():.3f}")
    optimizer.step()

# === 6. 测试模型 ===
model.eval()
likelihood.eval()

# 用更密集的测试点预测
test_x = torch.linspace(0, 1, 1000)
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-2):
    pred = likelihood(model(test_x))

# === 7. 可视化 ===
mean = pred.mean.numpy()
lower, upper = pred.confidence_region()
lower = lower.numpy()
upper = upper.numpy()

plt.figure(figsize=(10, 5))
plt.plot(x, y, label="True function", color='gray', alpha=0.6)
plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Train points', markersize=3)
plt.plot(test_x.numpy(), mean, 'b', label='GP Mean')
plt.fill_between(test_x.numpy(), lower, upper, alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title("Spectral GP Fit to $x\\sin(1000x)$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.grid(True)
plt.show()
