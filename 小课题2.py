import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# 1. 数据生成
train_x = torch.linspace(0, 1, 200)
train_y = (0.3 + 0.7 * train_x**2) * torch.sin(1500 * train_x)

# 标准化提高数值稳定性
train_y = (train_y - train_y.mean()) / train_y.std()

# 2. GP 模型定义
class SMGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = SpectralMixtureKernel(num_mixtures=10)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = GaussianLikelihood()
likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-4))

model = SMGPModel(train_x, train_y, likelihood)

# 3. 模型训练
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f"Iter {i+1}/100 - Loss: {loss.item():.3f}")
    optimizer.step()

# 4. 预测阶段
model.eval()
likelihood.eval()
test_x = torch.linspace(0, 1, 1000)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(test_x))
    mean = preds.mean
    lower, upper = preds.confidence_region()

# 5. 可视化
plt.figure(figsize=(10, 5))
plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Train Data')
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Predictive Mean')
plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='blue', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title("GP with Spectral Mixture Kernel Fit to $f(x) = (0.3 + 0.7x^2) \\cdot \\sin(1500x)$")
plt.grid(True)
plt.tight_layout()
plt.show()
