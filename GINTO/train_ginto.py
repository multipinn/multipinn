from multipinn.callbacks.curve import BasicCurve

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from multipinn.condition.condition import Condition
from multipinn.condition.diff import unpack
from multipinn.geometry.domain_nd import Hypercube
from multipinn.geometry.shell import Shell
from multipinn.generation.generator import Generator
from typing import Callable
import random
import torch
from torch import nn
import torch.distributed as dist
from poligons import polygons_5

polygons = polygons_5

class Shift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return x - self.shift


class SquareShift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return (x - self.shift) ** 2


class SigmoidShift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x - self.shift)


class ScaledTrig(nn.Module):
    def __init__(self, trig_func: Callable, scale: float):
        super().__init__()
        self.trig_func = trig_func
        self.scale = scale

    def forward(self, x):
        return self.trig_func(torch.pi * x * self.scale)


class MultiFeatureEncoding(nn.Module):
    def __init__(self, functions: nn.ModuleList = None):
        super().__init__()

        self.functions = (
            functions
            if functions
            else nn.ModuleList(
                [
                    nn.Identity(),
                    SquareShift(0),
                    SquareShift(1.5),
                    SquareShift(4.5),
                    SigmoidShift(0),
                    SigmoidShift(1.5),
                    SigmoidShift(4.5),
                    Shift(1.5),
                    Shift(4.5),
                    ScaledTrig(torch.sin, 1),
                    ScaledTrig(torch.cos, 1),
                    ScaledTrig(torch.sin, 1 / 4),
                    ScaledTrig(torch.cos, 1 / 4),
                    ScaledTrig(torch.sin, 1 / 2),
                    ScaledTrig(torch.cos, 1 / 2),
                ]
            )
        )

    @property
    def num_features(self):
        return len(self.functions)

    def forward(self, x):
        results = [f(x) for f in self.functions]
        return torch.hstack(results)


class PosEncoding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(MultiFeatureEncoding(), nn.Linear(30, out_dim))

    def forward(self, x):
        return torch.sin(self.fc(x))


class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, kernel_fn='relu'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        # для каждой головы своя матрица d_model -> head_dim
        self.q_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, kv):
        B, T_q, _ = q.shape

        q_heads = torch.stack([p(q) for p in self.q_projs], dim=1)
        k_heads = torch.stack([p(kv) for p in self.k_projs], dim=1)
        v_heads = torch.stack([p(kv) for p in self.v_projs], dim=1)

        q_heads = q_heads ** 2
        k_heads = k_heads ** 2

        kv_sum = torch.einsum('bhnd,bhne->bhde', k_heads, v_heads)
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q_heads, k_heads.sum(dim=2)) + 1e-6)
        attn_output = torch.einsum('bhnd,bhde,bhn->bhne', q_heads, kv_sum, z)

        # собрать обратно в (B, T_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(attn_output)


class CrossAttentionBlockLinear(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = LinearAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, q, kv):
        attn_output = self.attn(q, kv)
        x = q + attn_output
        x = x + self.ff(x)
        return x


class PINTO_2D(nn.Module):
    def __init__(self, m=128, num_heads=4, num_layers=3):
        super().__init__()
        self.pos_encoder = PosEncoding(out_dim=m)
        self.boundary_pos_encoder = PosEncoding(out_dim=m)
        self.cross_attn_blocks = nn.ModuleList([CrossAttentionBlockLinear(m, num_heads) for _ in range(num_layers)])
        self.projector = nn.Sequential(
            nn.Linear(m,m//2),
            nn.GELU(),
            nn.Linear(m//2,3)
        )

    def forward(self, query_pts, boundary_pts):
        q_enc = self.pos_encoder(query_pts).unsqueeze(0)
        b_pos_enc = self.boundary_pos_encoder(boundary_pts).unsqueeze(0)
        h = q_enc
        for block in self.cross_attn_blocks:
            h = block(h, b_pos_enc)
        out = self.projector(h)
        return out.squeeze(0)

def physics_loss_ns(
    model, 
    domain_pts,
    block_pts,
    boundary_pts,
    boundary_vals,
    boundary_inds,
    boundary_weight
):
    f = model(domain_pts, block_pts)
    argx_plus = domain_pts.clone()
    argx_minus = domain_pts.clone()
    argy_plus = domain_pts.clone()
    argy_minus = domain_pts.clone()
    argx_plus[:, 0] += 1.e-2
    argx_minus[:, 0] -= 1.e-2
    argy_plus[:, 1] += 1.e-2
    argy_minus[:, 1] -= 1.e-2
    fx_plus = model(argx_plus.clone(), block_pts)
    fx_minus = model(argx_minus.clone(), block_pts)
    fy_plus = model(argy_plus.clone(), block_pts)
    fy_minus = model(argy_minus.clone(), block_pts)

    f_x = (fx_plus - fx_minus) / 2.e-2
    f_y = (fy_plus - fy_minus) / 2.e-2
    f_xx = (fx_plus - 2 * f + fx_minus) / 1.e-2 ** 2
    f_yy = (fy_plus - 2 * f + fy_minus) / 1.e-2 ** 2

    u, v, _ = unpack(f)
    u_x, v_x, p_x = unpack(f_x)
    u_y, v_y, p_y = unpack(f_y)
    u_xx, v_xx, _ = unpack(f_xx)
    u_yy, v_yy, _ = unpack(f_yy)

    laplace_u = u_xx + u_yy
    laplace_v = v_xx + v_yy
    eq1 = u * u_x + v * u_y + p_x - 1.0 / 50.0 * laplace_u
    eq2 = u * v_x + v * v_y + p_y - 1.0 / 50.0 * laplace_v
    eq3 = u_x + v_y

    pde_residual = eq1.pow(2).mean() + eq2.pow(2).mean() + eq3.pow(2).mean()
    
    u_boundary_pred = model(boundary_pts, block_pts)
    bc_loss = (boundary_weight * (u_boundary_pred[torch.arange(len(boundary_inds)),boundary_inds].reshape(-1,1) - boundary_vals).pow(2)).mean()
    
    return pde_residual, bc_loss

def get_domain(obstacle, device):
    print("Определяем области")
    rect_back = Hypercube([-5, -2.5], [5, 2.5])
    domain = rect_back - obstacle

    x_min = Hypercube(low=[-5.0, -2.5], high=[-5.0, 2.5])
    x_max = Hypercube(low=[5.0, -2.5], high=[5.0, 2.5])
    y_min = Hypercube(low=[-5.0, -2.5], high=[5.0, -2.5])
    y_max = Hypercube(low=[-5.0, 2.5], high=[5.0, 2.5])

    shell = Shell(domain)
    block = shell - (x_min | x_max | y_min | y_max)
    walls = shell - (x_min | x_max | obstacle)

    c_domain = Condition(lambda x: x, domain)
    c_min = Condition(lambda x: x, x_min)
    c_max = Condition(lambda x: x, x_max)
    c_block = Condition(lambda x: x, block)
    c_walls = Condition(lambda x: x, walls)

    conds = [c_domain,  c_min, c_max, c_walls, c_block]

    generator_domain = Generator(
        n_points=100_000, sampler="Hammersley"
    )
    generator_bound = Generator(
        n_points=5000, sampler="Hammersley"
    )
    generator_walls = Generator(
        n_points=5000, sampler="Hammersley"
    )
    generator_block = Generator(
        n_points=1000, sampler="Hammersley"
    )

    print("Генерим точки")
    generator_domain.use_for(c_domain)
    generator_bound.use_for(c_min)
    generator_bound.use_for(c_max)
    generator_walls.use_for(c_walls)
    generator_block.use_for(c_block)

    for c in conds:
        c.update_points()

    domain_pts = c_domain.points.to(device)
    block_pts = c_block.points.to(device)
    
    print("Собираем все вместе")
    boundary_pts = torch.cat([c_min.points, c_min.points.clone(), c_walls.points, c_walls.points.clone(), c_block.points, c_block.points.clone(), c_max.points], dim=0).to(device).detach()
    boundary_vals = torch.cat([-0.16 * c_min.points[:,1].clone()**2 + 1, 
                               torch.zeros(c_min.points.shape[0]),
                               torch.zeros(c_walls.points.shape[0]), 
                               torch.zeros(c_walls.points.shape[0]),
                               torch.zeros(c_block.points.shape[0]), 
                               torch.zeros(c_block.points.shape[0]),
                               torch.zeros(c_max.points.shape[0])
                               ], dim=0).reshape(-1,1).to(device).detach()
    boundary_inds = torch.cat([torch.ones(c_min.points.shape[0]) * 0, 
                               torch.ones(c_min.points.shape[0]) * 1,
                               torch.ones(c_walls.points.shape[0]) * 1, 
                               torch.ones(c_walls.points.shape[0]) * 0,
                               torch.ones(c_block.points.shape[0]) * 1, 
                               torch.ones(c_block.points.shape[0]) * 0,
                               torch.ones(c_max.points.shape[0]) * 2
                               ], dim=0).tolist()
    boundary_weight = torch.cat([torch.ones(c_min.points.shape[0]) * 1, 
                        torch.ones(c_min.points.shape[0]) * 1,
                        torch.ones(c_walls.points.shape[0]) * 1, 
                        torch.ones(c_walls.points.shape[0]) * 1,
                        torch.ones(c_block.points.shape[0]) * 1, 
                        torch.ones(c_block.points.shape[0]) * 1,
                        torch.ones(c_max.points.shape[0]) * 1
                        ], dim=0).to(device)
    boundary_weight.requires_grad = False

    return domain_pts, block_pts, boundary_pts, boundary_vals, boundary_inds, boundary_weight, 

def select_accumulation_batch(domain_pts, block_pts, boundary_pts, boundary_vals, boundary_inds, boundary_weight, i, accumulation_steps):
    domain_pts_batch = domain_pts[i * domain_pts.shape[0] // accumulation_steps : (i + 1) * domain_pts.shape[0] // accumulation_steps]
    block_pts_batch = block_pts[i * block_pts.shape[0] // accumulation_steps : (i + 1) * block_pts.shape[0] // accumulation_steps]
    boundary_pts_batch = boundary_pts[i * boundary_pts.shape[0] // accumulation_steps : (i + 1) * boundary_pts.shape[0] // accumulation_steps]
    boundary_vals_batch = boundary_vals[i * boundary_vals.shape[0] // accumulation_steps : (i + 1) * boundary_vals.shape[0] // accumulation_steps]
    boundary_inds_batch = boundary_inds[i * len(boundary_inds) // accumulation_steps : (i + 1) * len(boundary_inds) // accumulation_steps]
    boundary_weight_batch = boundary_weight[i * boundary_weight.shape[0] // accumulation_steps : (i + 1) * boundary_weight.shape[0] // accumulation_steps]
    return domain_pts_batch, block_pts_batch, boundary_pts_batch, boundary_vals_batch, boundary_inds_batch, boundary_weight_batch

def group_tensors(data, group_size=10):
    n = len(data)
    m = len(data[0])
    result = []
    for i in range(0, n, group_size):
        group = data[i: i + group_size]
        merged = []
        for j in range(m):
            elems = [g[j] for g in group]
            if isinstance(elems[0], torch.Tensor):
                merged.append(torch.cat(elems, dim=0))
            elif isinstance(elems[0], list):
                merged.append(sum(elems, []))
            else:
                raise TypeError(f"Unsupported type {type(elems[0])}")
        result.append(tuple(merged))
    return result

def train_2d_ns():
    dist.init_process_group(backend='nccl')

    seed = 256
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    num_gpus = int(os.environ.get('NUM_GPUS_TRAIN', dist.get_world_size()))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    model = PINTO_2D().to(device)

    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    train_obscales = polygons
    obstacles_per_node = len(train_obscales) // dist.get_world_size()
    train_domains = [get_domain(obstacle, device) for obstacle in train_obscales[rank*obstacles_per_node:rank*obstacles_per_node+obstacles_per_node]]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    
    list_pde_loss, list_bs_loss, list_loss, list_lr = [], [], [], []

    if rank == 0:
        default_style = {
            "layout_yaxis_type": "log",
            "layout_title": "Loss curve",
            "layout_xaxis_title": "Epoch",
            "layout_yaxis_title": "Loss",
        }

        curve_logger = BasicCurve(
            save_dir="results/loss",     
            period=1000,           
            save_name="loss",   
            save_mode="html",
            style=default_style,            
        )

        lr_logger = BasicCurve(
            save_dir="5_art/lr",     
            period=1000,           
            save_name="lr",   
            save_mode="html",
            style={
                "layout_yaxis_type": "linear",
                "layout_title": "Learning Rate Curve",
                "layout_xaxis_title": "Epoch",
                "layout_yaxis_title": "Learning Rate",
            },            
        )
    
    pbar = tqdm(range(100_000), desc="Training")
    for epoch in pbar:
        running_pde_loss = 0.0
        running_bc_loss = 0.0
        optimizer.zero_grad()
        for domain_pts, block_pts, boundary_pts, boundary_vals, boundary_inds, boundary_weight in train_domains:
            pde_residual, bc_loss = physics_loss_ns(
                model,
                domain_pts, block_pts, boundary_pts, boundary_vals, boundary_inds, boundary_weight,
            )

            ((pde_residual + bc_loss) / obstacles_per_node).backward()

            running_pde_loss += pde_residual.detach() / obstacles_per_node
            running_bc_loss += bc_loss.detach() / obstacles_per_node
        
        size = float(dist.get_world_size())
        dist.all_reduce(running_pde_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_bc_loss, op=dist.ReduceOp.SUM)
        running_pde_loss /= size
        running_bc_loss /= size

        running_pde_loss = running_pde_loss.item()
        running_bc_loss = running_bc_loss.item()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step() 

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            list_lr.append(current_lr)

            pbar.set_postfix({
                'loss': f'{(running_pde_loss + running_bc_loss):.6f}',
                'PDE Loss': f'{running_pde_loss:.6f}',
                'BC Loss': f'{running_bc_loss:.6f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
            list_pde_loss.append(running_pde_loss)
            list_bs_loss.append(running_bc_loss)
            list_loss.append(running_pde_loss + running_bc_loss)

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f"results/mod/mod_{epoch}.pth")

            if epoch % curve_logger.period == 0 and epoch != 0:
                curve_logger.draw(
                    values=[list_pde_loss, list_bs_loss, list_loss],
                    v_names=["PDE Loss", "BC Loss", "Total Loss"],
                    coord=np.arange(len(list_loss))
                )

            if epoch % lr_logger.period == 0 and epoch != 0:
                lr_logger.draw(
                    values=[list_lr],
                    v_names=["Learning Rate"],
                    coord=np.arange(len(list_lr))
                )
    dist.destroy_process_group()

if __name__=="__main__":
    print("Start")
    train_2d_ns()
