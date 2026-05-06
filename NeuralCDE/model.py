import torch
import torchcde
import torch.nn as nn

class CDEFunc(nn.Module):
    """
    定义微分方程 dz/dt = f(z) dX/dt 中的 f(z)
    """
    def __init__(self, args):
        super(CDEFunc, self).__init__()
        self.input_channels = args.var_num + 1  # 加上时间通道
        self.hidden_channels = args.model_dim
        
        # 一个简单的 MLP，将隐藏状态映射为导数矩阵
        self.linear1 = nn.Linear(self.hidden_channels, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, self.hidden_channels * self.input_channels)

    def forward(self, t, z):
        # z shape: (batch, hidden_channels)
        z = self.linear1(z)
        z = self.relu(z)
        z = self.linear2(z)
        
        # 为了进行矩阵乘法，必须 reshape 成 (batch, hidden, input)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        
        # 这里的 tanh 是为了限制梯度的幅度，有助于 ODE 求解的稳定性
        return torch.tanh(z)

class NeuralCDE(nn.Module):
    def __init__(self, args):
        super(NeuralCDE, self).__init__()
        self.args = args
        self.input_channels = args.var_num + 1  # 加上时间通道
        self.hidden_channels = args.model_dim
        self.output_channels = args.var_num  # 预测目标的维度
        
        
        # 1. 向量场函数
        self.func = CDEFunc(args)
        
        # 2. 初始状态映射：将 t=0 的原始数据映射到隐藏空间 z0
        self.initial = nn.Linear(self.input_channels, self.hidden_channels)
        
        # 3. 读出层：将最终状态 zT 映射到预测目标
        self.readout = nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, coeffs):
        # 从系数构建连续路径
        X = torchcde.CubicSpline(coeffs)
        
        # 获取 t=0 时刻的数据作为初始条件
        x0 = X.evaluate(X.interval[0])
        z0 = self.initial(x0)
        
        seq_len = self.args.seq_len
        t_seq = torch.linspace(X.interval[0], X.interval[1], seq_len).to(z0.device)

        # 积分求解 (Integration)
        # adjoint=False 速度快但显存高，adjoint=True 显存低但速度慢
        # 返回的是 [z_t0, z_t1]，我们只需要最后一个时间点的状态
        z_T = torchcde.cdeint(X=X, 
                              func=self.func, 
                              z0=z0, 
                              t=t_seq,
                              atol=1e-3,
                              rtol=1e-3)
        
        # 取序列的最后一个状态 (Batch, Hidden)
        # z_T = z_T[:, -1]
        
        # 预测
        return self.readout(z_T)