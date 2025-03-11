import torch
import torch.nn as nn
from torch_scatter import scatter

# -----------------------------
# 1. Упрощённый класс Attention
#    (возвращает случайные attention-веса для демонстрации)
# -----------------------------
class SpGraphTransAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, opt, device, edge_weights=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.opt = opt
        # Для примера положим edge_weights в self, но не будем особо использовать
        self.edge_weights = edge_weights

    def forward(self, x, edge_index):
        # Возвращаем случайный attention-тензор размера [num_edges, 1]
        att = torch.rand(edge_index.size(1), 1, device=x.device, requires_grad=True)
        return att, None


# -----------------------------
# 2. Упрощённый класс ODEFunc
#    (только для демонстрации - ничего "умного" не делает)
# -----------------------------
class DummyODEFunc(nn.Module):
    def __init__(self, in_features, out_features, opt, data, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.opt = opt
        self.device = device

        # Параметры-пример:
        self.alpha_train = nn.Parameter(torch.tensor(1.0))
        self.beta_train  = nn.Parameter(torch.tensor(0.5))

        # Инициализируем пустые edge_index/weights
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None

        # Счётчик вызовов
        self.nfe = 0
        self.max_nfe = 100

    def forward(self, t, x):
        # Самая простая "ODE": f = alpha*(W*x - x) + beta
        self.nfe += 1
        if self.nfe > self.max_nfe:
            raise RuntimeError("Too many function evaluations")

        # Просто какая-нибудь "операция" – здесь для демонстрации
        # не используем self.edge_weight / self.attention_weights
        return self.alpha_train * x - x + self.beta_train


# -----------------------------
# 3. Класс-родитель ODEblock (упрощённая версия)
# -----------------------------
class ODEblock(nn.Module):
    def __init__(self, odefunc, regularization_fns, opt, data, device, t):
        super().__init__()
        self.odefunc_builder = odefunc
        self.regularization_fns = regularization_fns
        self.opt = opt
        self.data = data
        self.device = device
        self.t = t
        self.nreg = 0   # По умолчанию нет регуляризаторов

    def set_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-7
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-7


# -----------------------------
# 4. Реализация HardAttODEblock (из вашего кода, чуть упрощён)
# -----------------------------
class HardAttODEblock(ODEblock):
    def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1])):
        super().__init__(odefunc, regularization_fns, opt, data, device, t)
        assert 0 < opt['att_samp_pct'] <= 1, "attention sampling threshold must be in (0,1]"

        # Создаём "внутреннюю" odefunc
        self.odefunc = self.odefunc_builder(
            in_features = opt['hidden_dim'],
            out_features= opt['hidden_dim'],
            opt=opt, data=data, device=device
        )

        self.num_nodes = data.num_nodes

        # Допустим, edge_index & edge_weight делаем фиктивными
        if data.edge_index is None:
            # Создадим какой-нибудь тривиальный список рёбер
            edges = []
            for i in range(data.num_nodes):
                for j in range(data.num_nodes):
                    if i != j:
                        edges.append((i, j))
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_weight = torch.ones(edge_index.size(1))
        else:
            edge_index = data.edge_index
            edge_weight = data.edge_attr
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1))

        self.data_edge_index = edge_index.to(device)
        self.odefunc.edge_index = edge_index.to(device)
        self.odefunc.edge_weight = edge_weight.to(device)

        self.set_tol()

        # Показываем пример, что attention можно завести:
        self.multihead_att_layer = SpGraphTransAttentionLayer(
            in_features=opt['hidden_dim'],
            out_features=opt['hidden_dim'],
            opt=opt, device=device,
            edge_weights=self.odefunc.edge_weight
        ).to(device)

        # Выберем любой интегратор (rk4)
        from torchdiffeq import odeint
        self.train_integrator = odeint
        self.test_integrator = odeint

    def get_attention_weights(self, x):
        # Для примера – просто вызываем multihead_att_layer
        attention, values = self.multihead_att_layer(x, self.data_edge_index)
        return attention

    def renormalise_attention(self, attention):
        # Пример вашей renormalise
        index = self.odefunc.edge_index[self.opt['attention_norm_idx']]
        att_sums = scatter(attention, index, dim=0, dim_size=self.num_nodes, reduce='sum')[index]
        return attention / (att_sums + 1e-16)

    def forward(self, x):
        t = self.t.type_as(x)

        # 1) Считаем attention
        attention_weights = self.get_attention_weights(x)

        # 2) Если train, сделаем "фильтрацию" рёбер
        if self.training:
            with torch.no_grad():
                mean_att = attention_weights.mean(dim=1, keepdim=False)
                threshold = torch.quantile(mean_att, 1-self.opt['att_samp_pct'])
                mask = mean_att > threshold
                self.odefunc.edge_index = self.data_edge_index[:, mask]
                sampled_attention_weights = self.renormalise_attention(mean_att[mask])
                self.odefunc.attention_weights = sampled_attention_weights
        else:
            self.odefunc.edge_index = self.data_edge_index
            # Возьмём просто среднее внимание
            self.odefunc.attention_weights = attention_weights.mean(dim=1, keepdim=False)

        # 3) Вызываем torchdiffeq (rk4)
        # Здесь для упрощения считаем, что nreg=0
        state_dt = self.train_integrator(
            self.odefunc,  # функция
            x,             # начальное состояние
            t,             # времена [t0, t1] = [0,1]
            method='rk4',
            atol=self.atol, rtol=self.rtol
        )

        # state_dt.shape: [2, batch_size, dim]
        # state_dt[0] = x(t=0), state_dt[1] = x(t=1)
        z = state_dt[1]

        # Вернём решение в момент времени 1
        return z


# -----------------------------
# 5. Простая проверка
# -----------------------------
if __name__ == '__main__':
    # 5.1. Создадим dummy-данные
    class DummyData:
        def __init__(self, num_nodes, dim):
            self.num_nodes = num_nodes
            self.x = torch.randn(num_nodes, dim)
            self.edge_index = None
            self.edge_attr = None
            # Можно заполнить edge_index, edge_attr, если хотите

    data = DummyData(num_nodes=5, dim=8)

    # 5.2. Настройки (opt)
    opt = {
        'att_samp_pct': 0.5,
        'hidden_dim': 8,
        'attention_norm_idx': 0,
        'use_flux': False,
        'self_loop_weight': 1.0,
        'function': 'laplacian',   # не 'transformer'/'GAT'
    }

    device = torch.device('cpu')
    data.x = data.x.to(device)

    # 5.3. Создаём блок HardAttODEblock
    #     (при этом внутри создаётся DummyODEFunc)
    block = HardAttODEblock(
        odefunc=DummyODEFunc,
        regularization_fns=[],
        opt=opt,
        data=data,
        device=device,
        t=torch.tensor([0,5], dtype=torch.float)
    ).to(device)

    # 5.4. Пробуем прогнать вперёд и назад
    block.train()  # включаем режим train, чтобы сработал "attention sampling"

    x_in = data.x.clone().detach().requires_grad_(True)
    z_out = block(x_in)

    print("z_out.shape =", z_out.shape)
    print("z_out.requires_grad =", z_out.requires_grad)

    # 5.5. Считаем фейковый лосс и делаем backward
    loss = z_out.sum()
    loss.backward()

    # 5.6. Смотрим градиенты у параметров ODEFunc
    print("\nПараметры ODEFunc и их градиенты:")
    for name, param in block.odefunc.named_parameters():
        print(f"{name} -> grad = {param.grad}")

    # А также градиент по входу (x_in)
    print("\nГрадиент по входу x_in:")
    print(x_in.grad)
