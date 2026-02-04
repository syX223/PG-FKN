import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import median_filter

plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["lines.linewidth"] = 3.0
plt.rcParams["lines.markersize"] = 8
plt.rcParams["figure.dpi"] = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
np.random.seed(66)
torch.manual_seed(66)

config = {
    "lon_min": 112.0, "lon_max": 118.0,
    "lat_min": 12.0, "lat_max": 18.0,
    "depth_min": 0.0, "depth_max": 1000.0,
    "time_min": 2023.0, "time_max": 2023.25,
    "lon_grid_num": 50, "lat_grid_num": 50, "depth_grid_num": 30,
    "N_pde": 8000, "N_pde_thermo": 5000,
    "N_pde_surface": 2500,
    "N_bc": 5000,
    "hidden_dim": 50, "hidden_layers": 4, "lr_adam": 5e-4, "lr_lbfgs": 5e-4,
    "epochs_adam": 30000, "epochs_lbfgs": 500,
    "lambda_pde_init": 0.01, "lambda_pde_final": 10.0, "lambda_data": 10.0,
    "lambda_bc": 1.0, "lambda_reg": 0.02,
    "woa_temp_path": " ",
    "woa_sal_path": " ",
    "argo_local_path": " ",
    "model_save_path": " ",
    "fig_save_dir": " "
}
os.makedirs(config["fig_save_dir"], exist_ok=True)

def load_woa_data():
    ds_temp = xr.open_dataset(config["woa_temp_path"])["t_an"]
    ds_sal = xr.open_dataset(config["woa_sal_path"])["s_an"]
    woa_temp = ds_temp.sel(
        lon=slice(config["lon_min"], config["lon_max"]),
        lat=slice(config["lat_min"], config["lat_max"]),
        depth=slice(config["depth_min"], config["depth_max"])
    )
    woa_sal = ds_sal.sel(
        lon=slice(config["lon_min"], config["lon_max"]),
        lat=slice(config["lat_min"], config["lat_max"]),
        depth=slice(config["depth_min"], config["depth_max"])
    )
    lon_grid = np.linspace(config["lon_min"], config["lon_max"], config["lon_grid_num"])
    lat_grid = np.linspace(config["lat_min"], config["lat_max"], config["lat_grid_num"])
    depth_grid = np.linspace(config["depth_min"], config["depth_max"], config["depth_grid_num"])
    woa_temp_interp = woa_temp.interp(lon=lon_grid, lat=lat_grid, depth=depth_grid).values
    woa_sal_interp = woa_sal.interp(lon=lon_grid, lat=lat_grid, depth=depth_grid).values
    for k in range(woa_temp_interp.shape[0]):
        layer = woa_temp_interp[k, :, :]
        if np.isnan(layer).sum() == 0:
            continue
        lon_idx, lat_idx = np.where(~np.isnan(layer))
        lon_vals = lon_grid[lon_idx]
        lat_vals = lat_grid[lat_idx]
        val_vals = layer[lon_idx, lat_idx]
        if len(val_vals) < 3:
            print(f"温度场第{k}层有效点不足3个，使用3×3中值滤波 fallback")
            layer = median_filter(layer, size=3, mode='nearest')
        else:
            ok = OrdinaryKriging(lon_vals, lat_vals, val_vals, variogram_model="spherical")
            z, _ = ok.execute("grid", lon_grid, lat_grid)
            layer = z
        woa_temp_interp[k, :, :] = layer
    for k in range(woa_sal_interp.shape[0]):
        layer = woa_sal_interp[k, :, :]
        if np.isnan(layer).sum() == 0:
            continue
        lon_idx, lat_idx = np.where(~np.isnan(layer))
        lon_vals = lon_grid[lon_idx]
        lat_vals = lat_grid[lat_idx]
        val_vals = layer[lon_idx, lat_idx]
        if len(val_vals) < 3:
            print(f"盐度场第{k}层有效点不足3个，使用3×3中值滤波 fallback")
            layer = median_filter(layer, size=3, mode='nearest')
        else:
            ok = OrdinaryKriging(
                lon_vals, lat_vals, val_vals,
                variogram_model="spherical",
                verbose=False,
                enable_plotting=False
            )
            z, _ = ok.execute("grid", lon_grid, lat_grid)
            layer = z
        woa_sal_interp[k, :, :] = layer
    total_grid = lon_grid.size * lat_grid.size * depth_grid.size
    temp_nan_count = np.isnan(woa_temp_interp).sum()
    sal_nan_count = np.isnan(woa_sal_interp).sum()
    valid_temp = total_grid - temp_nan_count
    valid_sal = total_grid - sal_nan_count
    print(f"WOA数据实际覆盖范围：")
    print(f"  经度：{woa_temp.lon.min().item():.2f}~{woa_temp.lon.max().item():.2f}")
    print(f"  纬度：{woa_temp.lat.min().item():.2f}~{woa_temp.lat.max().item():.2f}")
    print(f"  深度：{woa_temp.depth.min().item():.2f}~{woa_temp.depth.max().item():.2f}")
    print(f"WOA数据预处理完成：")
    print(f"  总网格数：{total_grid}（经度{lon_grid.size}×纬度{lat_grid.size}×深度{depth_grid.size}）")
    print(f"  温度场：有效数据{valid_temp}条（填充后缺失{temp_nan_count}条）")
    print(f"  盐度场：有效数据{valid_sal}条（填充后缺失{sal_nan_count}条）")
    global c_true_woa
    z_mesh = np.tile(depth_grid.reshape(-1, 1, 1), (1, config["lat_grid_num"], config["lon_grid_num"]))
    c_true_woa = mackenzie(woa_temp_interp, woa_sal_interp, z_mesh)  # (depth, lat, lon)
    c_true_woa_flat = c_true_woa.ravel()
    global c_true_tensor
    c_true_tensor = torch.tensor(c_true_woa_flat, dtype=torch.float32).to(device)
    print(f"WOA声速范围：{c_true_woa.min():.2f} ~ {c_true_woa.max():.2f} m/s")
    global c_woa_interpolator
    c_woa_interpolator = RegularGridInterpolator(
        points=(depth_grid, lat_grid, lon_grid),
        values=c_true_woa,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    return woa_temp_interp, woa_sal_interp, lon_grid, lat_grid, depth_grid

def load_argo_data():
    argo_files = [
        os.path.join(config["argo_local_path"], " "),
        os.path.join(config["argo_local_path"], " "),
        os.path.join(config["argo_local_path"], " ")
    ]
    for file in argo_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Argo文件不存在：{file}\n请检查文件路径或名称")
    core_vars = {
        "lon": "lon",
        "lat": "lat",
        "pres": "pres",
        "temp": "temp",
        "salt": "salt",
        "temp_scatter_error": "temp_scatter_error",
        "salt_scatter_error": "salt_scatter_error",
        "time": "time"
    }
    argo_dfs = []
    for nc_file in argo_files:
        try:
            try:
                ds = xr.open_dataset(nc_file)
            except:
                ds = xr.open_dataset(nc_file, decode_times=False)
            missing_vars = [v for v in core_vars.values() if v not in ds.variables]
            if missing_vars:
                print(f"跳过文件{nc_file}：缺失变量{missing_vars}")
                continue
            df = ds[list(core_vars.values())].to_dataframe().reset_index()
            df = df.rename(columns={v: k for k, v in core_vars.items()})
            argo_dfs.append(df)
        except Exception as e:
            print(f"读取文件{nc_file}失败：{str(e)}")
            continue
    if len(argo_dfs) == 0:
        raise ValueError("未读取到任何有效Argo数据，请检查文件格式或变量完整性")
    argo_df = pd.concat(argo_dfs, ignore_index=True)
    print(f"本地Argo数据读取完成：共{len(argo_df)}条原始记录")
    if "time" in argo_df.columns:
        base_date = datetime(1, 1, 1)
        argo_df["TIME"] = argo_df["time"].apply(
            lambda x: base_date + timedelta(days=float(x)) - timedelta(days=365)
        )
        time_start = datetime(2023, 1, 1)
        time_end = datetime(2023, 3, 31)
        argo_df = argo_df[(argo_df["TIME"] >= time_start) & (argo_df["TIME"] <= time_end)]
        if len(argo_df) == 0:
            raise ValueError(f"时间筛选后无数据（{time_start.strftime('%Y-%m-%d')}至{time_end.strftime('%Y-%m-%d')}）")
    else:
        print("警告：Argo文件无time变量，时间筛选逻辑已跳过！")
    woa_lon_min, woa_lon_max = config["lon_min"], config["lon_max"]
    woa_lat_min, woa_lat_max = config["lat_min"], config["lat_max"]
    woa_depth_min, woa_depth_max = config["depth_min"], config["depth_max"]
    buffer_lon = 0.01 * (woa_lon_max - woa_lon_min)
    buffer_lat = 0.01 * (woa_lat_max - woa_lat_min)
    buffer_depth = 0.01 * (woa_depth_max - woa_depth_min)
    argo_surface = argo_df[
        (argo_df["lon"] >= woa_lon_min) & (argo_df["lon"] <= woa_lon_max) &
        (argo_df["lat"] >= woa_lat_min) & (argo_df["lat"] <= woa_lat_max) &
        (argo_df["pres"] >= woa_depth_min) & (argo_df["pres"] <= 50.0)
        ].copy()
    argo_non_surface = argo_df[
        (argo_df["lon"] >= woa_lon_min + buffer_lon) & (argo_df["lon"] <= woa_lon_max - buffer_lon) &
        (argo_df["lat"] >= woa_lat_min + buffer_lat) & (argo_df["lat"] <= woa_lat_max - buffer_lat) &
        (argo_df["pres"] > 50.0)
        ].copy()
    argo_df = pd.concat([argo_surface, argo_non_surface], ignore_index=True)
    print(f"裁剪后Argo数据量（表层+非表层）：{len(argo_df)}")
    argo_df = argo_df[
        (argo_df["lon"] >= config["lon_min"]) & (argo_df["lon"] <= config["lon_max"]) &
        (argo_df["lat"] >= config["lat_min"]) & (argo_df["lat"] <= config["lat_max"]) &
        (argo_df["pres"] >= config["depth_min"]) & (argo_df["pres"] <= config["depth_max"])
        ].copy()
    if len(argo_df) == 0:
        raise ValueError("区域筛选后无数据，请检查经纬度/深度范围与数据匹配度")
    argo_clean = argo_df[
        (~argo_df["temp_scatter_error"].isna()) &
        (~argo_df["salt_scatter_error"].isna()) &
        # 表层（0-50m）误差更严格，非表层放宽
        ( (argo_df["pres"] <=50) & (argo_df["temp_scatter_error"] < 0.3) & (argo_df["salt_scatter_error"] < 0.1) ) |
        ( (argo_df["pres"] >50) & (argo_df["temp_scatter_error"] < 1.0) & (argo_df["salt_scatter_error"] < 0.5) )
    ].copy()
    argo_clean = argo_clean.dropna(subset=["lon", "lat", "pres", "temp", "salt"])
    if len(argo_clean) == 0:
        print("警告：严格误差阈值后无数据，尝试放宽条件...")
        argo_clean = argo_df[
            (~argo_df["temp_scatter_error"].isna()) &
            (~argo_df["salt_scatter_error"].isna()) &
            (argo_df["temp_scatter_error"] < 1.0) &
            (argo_df["salt_scatter_error"] < 0.5)
            ].copy()
        argo_clean = argo_clean.dropna(subset=["lon", "lat", "pres", "temp", "salt"])
    if len(argo_clean) == 0:
        print("警告：放宽阈值后仍无数据，仅排除NaN...")
        argo_clean = argo_df[
            (~argo_df["temp_scatter_error"].isna()) &
            (~argo_df["salt_scatter_error"].isna())
            ].copy()
        argo_clean = argo_clean.dropna(subset=["lon", "lat", "pres", "temp", "salt"])
    if len(argo_clean) == 0:
        raise ValueError("所有数据均为无效值（含NaN），请检查原始数据完整性")
    argo_clean = argo_clean.rename(columns={
        "lon": "LONGITUDE",
        "lat": "LATITUDE",
        "pres": "DEPTH",
        "temp": "TEMP",
        "salt": "PSAL",
        "temp_scatter_error": "TEMP_QC",
        "salt_scatter_error": "PSAL_QC"
    })
    argo_clean = argo_clean.dropna(subset=["LONGITUDE", "LATITUDE", "DEPTH", "TEMP", "PSAL"])
    print(f"Argo数据预处理完成：{len(argo_clean)}条有效数据")
    return argo_clean

def mackenzie(temp, sal, depth):
    c = 1448.96 + 4.591 * temp - 5.304e-2 * temp ** 2 + 2.374e-4 * temp ** 3 + \
        (1.340 * (sal - 35)) + 1.630e-2 * depth + 1.675e-7 * depth ** 2 - \
        1.025e-2 * temp * (sal - 35) - 7.139e-13 * temp * depth ** 3
    return c

def normalize_coord(coord, min_val, max_val):
    return (coord - min_val) / (max_val - min_val)

def denormalize_coord(norm_val, min_val, max_val):
    return norm_val * (max_val - min_val) + min_val

def generate_pde_points(woa_temp, woa_sal, lon_grid, lat_grid, depth_grid):
    lhs = LatinHypercube(d=3, seed=66)
    pde_norm = lhs.random(n=config["N_pde"])
    sal_diff = np.abs(np.diff(woa_sal, axis=0))
    temp_diff = np.abs(np.diff(woa_temp, axis=0))
    depth_diff = np.diff(depth_grid)
    depth_diff_reshaped = depth_diff.reshape(-1, 1, 1)
    temp_grad = temp_diff / depth_diff_reshaped
    temp_grad_per_depth = temp_grad.mean(axis=(1, 2))
    sal_diff = np.abs(np.diff(woa_sal, axis=0))
    sal_grad = sal_diff / depth_diff_reshaped
    sal_grad_per_depth = sal_grad.mean(axis=(1, 2))
    joint_grad_per_depth = (temp_grad_per_depth + sal_grad_per_depth) / 2
    joint_grad_weights = joint_grad_per_depth / joint_grad_per_depth.sum()
    joint_grad_weights = np.clip(joint_grad_weights, 1e-8, None)
    joint_grad_weights = joint_grad_weights / joint_grad_weights.sum()
    depth_indices = np.arange(len(joint_grad_weights))
    selected_depth_indices = np.random.choice(
        depth_indices,
        size=config["N_pde_thermo"],
        p=joint_grad_weights
    )
    thermo_depth = depth_grid[:-1][selected_depth_indices]
    thermo_depth += np.random.normal(0, 5, size=config["N_pde_thermo"])
    thermo_depth = np.clip(thermo_depth, config["depth_min"], config["depth_max"])
    thermo_lon = np.random.uniform(config["lon_min"], config["lon_max"], config["N_pde_thermo"])
    thermo_lat = np.random.uniform(config["lat_min"], config["lat_max"], config["N_pde_thermo"])
    thermo_norm = np.column_stack([
        normalize_coord(thermo_lon, config["lon_min"], config["lon_max"]),
        normalize_coord(thermo_lat, config["lat_min"], config["lat_max"]),
        normalize_coord(thermo_depth, config["depth_min"], config["depth_max"])
    ])
    n_surface = config["N_pde_surface"]
    n_surface_shallow = int(n_surface * 0.6)
    n_surface_deep = n_surface - n_surface_shallow
    surface_depth1 = np.random.uniform(0.0, 20.0, size=n_surface_shallow) + np.random.normal(0, 2,
                                                                                             size=n_surface_shallow)
    surface_depth2 = np.random.uniform(20.0, 50.0, size=n_surface_deep) + np.random.normal(0, 3, size=n_surface_deep)
    surface_depth = np.concatenate([surface_depth1, surface_depth2])
    surface_depth = np.clip(surface_depth, 0.0, 50.0)
    surface_lon = np.random.uniform(config["lon_min"], config["lon_max"], size=n_surface)
    surface_lat = np.random.uniform(config["lat_min"], config["lat_max"], size=n_surface)
    surface_norm = np.column_stack([
        normalize_coord(surface_lon, config["lon_min"], config["lon_max"]),
        normalize_coord(surface_lat, config["lat_min"], config["lat_max"]),
        normalize_coord(surface_depth, config["depth_min"], config["depth_max"])
    ])
    pde_norm = np.vstack([pde_norm, thermo_norm, surface_norm])  # 新增表层点合并
    time_min_norm = normalize_coord(config["time_min"], config["time_min"], config["time_max"])
    time_max_norm = normalize_coord(config["time_max"], config["time_min"], config["time_max"])
    time_norm = np.random.uniform(time_min_norm, time_max_norm, size=len(pde_norm))
    pde_norm = np.column_stack([pde_norm, time_norm])
    X_pde = torch.tensor(pde_norm, dtype=torch.float32, requires_grad=True).to(device)
    return X_pde

def generate_bc_points():
    N = config["N_bc"]
    area_sea = (config["lon_max"] - config["lon_min"]) * (config["lat_max"] - config["lat_min"])
    area_side = 4 * (config["lon_max"] - config["lon_min"]) * (config["depth_max"] - config["depth_min"])
    N_sea = int(N * area_sea / (2 * area_sea + area_side))
    N_side_per = (N - 2 * N_sea) // 4
    sea_surface = np.column_stack([
        np.random.uniform(0, 1, N_sea),
        np.random.uniform(0, 1, N_sea),
        np.zeros(N_sea),
        np.random.uniform(0, 1, N_sea)
    ])
    sea_surface_labels = np.zeros(N_sea)
    sea_bottom = np.column_stack([
        np.random.uniform(0, 1, N_sea),
        np.random.uniform(0, 1, N_sea),
        np.ones(N_sea),  # depth_norm=1
        np.random.uniform(0, 1, N_sea)
    ])
    sea_bottom_labels = np.ones(N_sea)
    side_xmin = np.column_stack([
        np.zeros(N_side_per),
        np.random.uniform(0, 1, N_side_per),
        np.random.uniform(0, 1, N_side_per),
        np.random.uniform(0, 1, N_side_per)
    ])
    side_xmax = np.column_stack([
        np.ones(N_side_per),
        np.random.uniform(0, 1, N_side_per),
        np.random.uniform(0, 1, N_side_per),
        np.random.uniform(0, 1, N_side_per)
    ])
    side_ymin = np.column_stack([
        np.random.uniform(0, 1, N_side_per),
        np.zeros(N_side_per),
        np.random.uniform(0, 1, N_side_per),
        np.random.uniform(0, 1, N_side_per)
    ])
    side_ymax = np.column_stack([
        np.random.uniform(0, 1, N_side_per),
        np.ones(N_side_per),
        np.random.uniform(0, 1, N_side_per),
        np.random.uniform(0, 1, N_side_per)
    ])
    side_labels = np.full(4 * N_side_per, 2)
    bc_norm = np.vstack([sea_surface, sea_bottom, side_xmin, side_xmax, side_ymin, side_ymax])
    bc_labels = np.concatenate([sea_surface_labels, sea_bottom_labels, side_labels])
    X_bc = torch.tensor(bc_norm, dtype=torch.float32, requires_grad=True).to(device)
    bc_labels = torch.tensor(bc_labels, dtype=torch.long).to(device)
    return X_bc, bc_labels

def generate_data_points(argo_clean, woa_temp, woa_sal, lon_grid, lat_grid, depth_grid):
    argo_lon_phys = argo_clean["LONGITUDE"].values
    argo_lat_phys = argo_clean["LATITUDE"].values
    argo_depth_phys = argo_clean["DEPTH"].values
    z_mesh = np.tile(
        depth_grid.reshape(-1, 1, 1),
        (1, config["lat_grid_num"], config["lon_grid_num"])
    )
    c_true = mackenzie(woa_temp, woa_sal, z_mesh)
    c_interp = RegularGridInterpolator(
        (depth_grid, lat_grid, lon_grid),
        c_true,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    argo_coords_phys = argo_clean[["DEPTH", "LATITUDE", "LONGITUDE"]].values
    c_argo = c_interp(argo_coords_phys)
    print(f"\nArgo声速（c_argo）原始NaN数量：{np.isnan(c_argo).sum()}")
    valid_mask = ~np.isnan(c_argo)
    argo_clean_valid = argo_clean[valid_mask].copy()
    c_argo_valid = c_argo[valid_mask]
    print(f"移除NaN后Argo有效数据量：{len(argo_clean_valid)}")
    p_true = 100 + 0.1 * c_argo_valid
    p_noisy = p_true + 0.01 * p_true.std() * np.random.randn(*p_true.shape)
    print(f"生成y_data：{len(p_noisy)}条，NaN数量：{np.isnan(p_noisy).sum()}")
    argo_lon_norm = normalize_coord(argo_clean_valid["LONGITUDE"].values, config["lon_min"], config["lon_max"])
    argo_lat_norm = normalize_coord(argo_clean_valid["LATITUDE"].values, config["lat_min"], config["lat_max"])
    argo_depth_norm = normalize_coord(argo_clean_valid["DEPTH"].values, config["depth_min"], config["depth_max"])
    argo_time_phys = np.random.uniform(config["time_min"], config["time_max"], size=len(argo_clean_valid))
    argo_time_norm = normalize_coord(argo_time_phys, config["time_min"], config["time_max"])
    data_norm = np.column_stack([argo_lon_norm, argo_lat_norm, argo_depth_norm, argo_time_norm])
    X_data = torch.tensor(data_norm, dtype=torch.float32).to(device)
    y_data = torch.tensor(p_noisy.reshape(-1, 1), dtype=torch.float32).to(device)
    print("y_data统计：")
    print(f"均值：{y_data.mean().item():.2f}，标准差：{y_data.std().item():.2f}")
    print(f"最小值：{y_data.min().item():.2f}，最大值：{y_data.max().item():.2f}")
    return X_data, y_data, p_true

class PINN(nn.Module):
    def __init__(self, input_dim=4, output_dim_p=1, output_dim_c=1):
        super(PINN, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module("input", nn.Linear(8, config["hidden_dim"]))
        self.layers.add_module("input_act", nn.Tanh())

        for i in range(config["hidden_layers"]):
            linear = nn.Linear(config["hidden_dim"], config["hidden_dim"])
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(linear.bias)
            self.layers.add_module(f"hidden_{i}", linear)
            self.layers.add_module(f"act_{i}", nn.Tanh())
        self.output_p = nn.Sequential(nn.Linear(config["hidden_dim"], output_dim_p))
        self.output_c = nn.Sequential(
            nn.Linear(config["hidden_dim"], 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        with torch.no_grad():
            nn.init.normal_(self.output_c[0].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.output_c[0].bias, 0.0)
            nn.init.normal_(self.output_c[2].weight, mean=0.0, std=0.001)  # 权重接近0，避免干扰初始值
            nn.init.constant_(self.output_c[2].bias, 1480.0)  # 强制初始偏置为1480m/s（南海典型声速）

    def forward(self, x):
        lon, lat, depth, time = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        surface_flag = (depth < 0.05).float()
        time_scaled = time * 0.5
        depth_scaled = depth * 5.0
        depth_time = depth_scaled * time_scaled
        x_scaled = torch.cat([
            lon, lat, depth_scaled, time_scaled,
            lon * time_scaled, lat * time_scaled, depth_time,
            surface_flag
        ], dim=1)
        hidden = self.layers(x_scaled)
        p = self.output_p(hidden)
        c = self.output_c(hidden)
        return p, c

    def compute_pde_residual(self, x, gamma=0.01):
        p, c = self(x)
        p_grad = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = p_grad[:, 0:1]
        p_y = p_grad[:, 1:2]
        p_z = p_grad[:, 2:3]
        p_t = p_grad[:, 3:4]
        p_x_grad = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
        p_y_grad = torch.autograd.grad(p_y, x, grad_outputs=torch.ones_like(p_y), create_graph=True)[0]
        p_z_grad = torch.autograd.grad(p_z, x, grad_outputs=torch.ones_like(p_z), create_graph=True)[0]
        p_t_grad = torch.autograd.grad(p_t, x, grad_outputs=torch.ones_like(p_t), create_graph=True)[0]
        p_xx = p_x_grad[:, 0:1]
        p_yy = p_y_grad[:, 1:2]
        p_zz = p_z_grad[:, 2:3]
        p_tt = p_t_grad[:, 3:4]
        laplacian_p = p_xx + p_yy + p_zz
        residual = p_tt - (c ** 2) * laplacian_p
        depth_phys = denormalize_coord(x[:,2:3], config["depth_min"], config["depth_max"])
        mixed_layer_mask = (depth_phys <= 20.0).float()
        if mixed_layer_mask.sum() > 0:
            c_grad = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
            mixed_layer_grad = c_grad * mixed_layer_mask.unsqueeze(1)
            residual += 10.0 * torch.mean(mixed_layer_grad ** 2)
        return residual

def compute_loss(model, X_pde, X_bc, bc_labels, X_data, y_data, lambda_pde, lambda_bc):
    loss_pde = torch.tensor(0.0, device=device)
    loss_data = torch.tensor(0.0, device=device)
    loss_bc = torch.tensor(0.0, device=device)
    loss_reg = torch.tensor(0.0, device=device)
    loss_bc_sea = torch.tensor(0.0, device=device)
    loss_bc_other = torch.tensor(0.0, device=device)
    loss_c = torch.tensor(0.0, device=device)
    c_range_penalty = torch.tensor(0.0, device=device)
    if torch.isnan(y_data).any():
        nan_count = torch.isnan(y_data).sum().item()
        print(f"⚠️ 数据标签y_data含{nan_count}个nan，已用0填充")
        y_data = torch.nan_to_num(y_data, nan=0.0)
    try:
        pde_res = model.compute_pde_residual(X_pde)
        if torch.isnan(pde_res).any():
            pde_res = torch.nan_to_num(pde_res, nan=0.0)
        loss_pde = torch.mean(pde_res ** 2)
    except Exception as e:
        print(f"❌ PDE残差计算失败：{e}")
    _, c_pde = model(X_pde)
    lon_pde_norm = X_pde[:, 0]
    lat_pde_norm = X_pde[:, 1]
    depth_pde_norm = X_pde[:, 2]
    lon_pde_phys = denormalize_coord(lon_pde_norm, config["lon_min"], config["lon_max"])
    lat_pde_phys = denormalize_coord(lat_pde_norm, config["lat_min"], config["lat_max"])
    depth_pde_phys = denormalize_coord(depth_pde_norm, config["depth_min"], config["depth_max"])
    pde_coords = torch.stack([depth_pde_phys, lat_pde_phys, lon_pde_phys], dim=1).detach().cpu().numpy()
    c_true_pde = c_woa_interpolator(pde_coords)
    valid_mask = ~np.isnan(c_true_pde)
    if np.sum(valid_mask) > 0:
        c_true_pde_valid = torch.tensor(c_true_pde[valid_mask], dtype=torch.float32).to(device).reshape(-1, 1)
        c_pde_valid = c_pde[torch.tensor(valid_mask, device=device)]
        loss_c = torch.mean((c_pde_valid - c_true_pde_valid) ** 2)
        c_min = torch.tensor(1450.0, device=device)
        c_max = torch.tensor(1550.0, device=device)
        c_range_penalty = torch.mean(torch.relu(c_min - c_pde_valid) + torch.relu(c_pde_valid - c_max))
    try:
        p_data, c_data = model(X_data)
        if torch.isnan(p_data).any():
            p_data = torch.nan_to_num(p_data, nan=0.0)
        depth_data_norm = X_data[:, 2]
        depth_data_phys = denormalize_coord(
            depth_data_norm,
            config["depth_min"],
            config["depth_max"]
        )
        data_weights = torch.where(
            depth_data_phys <= 20.0,
            torch.tensor(18.0, device=device),
            torch.where(
                depth_data_phys <= 50.0,
                torch.tensor(15.0, device=device),
                torch.tensor(5.0, device=device)
            )
        )
        loss_data = torch.mean(data_weights * (p_data - y_data) ** 2)
    except Exception as e:
        print(f"❌ 数据损失计算失败：{e}")
    try:
        p_bc, _ = model(X_bc)
        if torch.isnan(p_bc).any():
            p_bc = torch.nan_to_num(p_bc, nan=0.0)
        mask_sea = (bc_labels == 0)
        if mask_sea.any():
            X_sea = X_bc[mask_sea].detach().clone()
            X_sea.requires_grad = True
            p_sea, _ = model(X_sea)
            p_grad_sea = torch.autograd.grad(
                p_sea, X_sea,
                grad_outputs=torch.ones_like(p_sea, device=device),
                create_graph=True,
                retain_graph=True,
                allow_unused=False
            )[0]
            p_z_sea = p_grad_sea[:, 2:3]
            p_z_sea = torch.nan_to_num(p_z_sea, nan=0.0)
            loss_bc_sea = torch.mean(p_z_sea ** 2)
        mask_other = (bc_labels != 0)
        if mask_other.any():
            loss_bc_other = torch.mean(p_bc[mask_other] ** 2)
        loss_bc = loss_bc_sea + loss_bc_other
    except Exception as e:
        print(f"❌ 边界损失计算失败：{e}")
        loss_bc = torch.tensor(0.0, device=device)
    try:
        _, c_pde = model(X_pde)
        if torch.isnan(c_pde).any():
            c_pde = torch.nan_to_num(c_pde, nan=0.0)
        c_grad = torch.autograd.grad(c_pde, X_pde, grad_outputs=torch.ones_like(c_pde), create_graph=True)[0]
        loss_reg = torch.mean(c_grad ** 2)
    except Exception as e:
        print(f"❌ 正则化损失计算失败：{e}")
    total_loss = (
            lambda_pde * loss_pde +
            loss_data +
            lambda_bc * loss_bc +
            config["lambda_reg"] * loss_reg +
            150.0 * loss_c +
            50.0 * c_range_penalty
    )
    if torch.isnan(total_loss):
        print("⚠️ 总损失为nan，强制重置为1.0")
        total_loss = torch.tensor(1.0, device=device, requires_grad=True)
    return total_loss, (loss_pde.item(), loss_data.item(), loss_bc_sea.item(), loss_bc_other.item(), loss_reg.item(),
                        loss_c.item())

def train_model(model, X_pde, X_bc, bc_labels, X_data, y_data):
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=config["lr_adam"])
    scheduler = CosineAnnealingLR(optimizer_adam, T_max=config["epochs_adam"], eta_min=1e-6)
    print("=== Adam预训练（动态调整BC权重） ===")
    total_adam_epochs = config["epochs_adam"]
    for epoch in tqdm(range(total_adam_epochs), desc="Adam训练进度"):
        # 动态调整lambda_pde和lambda_bc
        if epoch < total_adam_epochs * 0.25:
            lambda_pde = 0.01
            lambda_bc = 50.0
        elif epoch < total_adam_epochs * 0.5:
            lambda_pde = 0.1
            lambda_bc = 20.0
        else:
            lambda_pde = 0.1 + (1.0 - 0.1) * ((epoch - total_adam_epochs * 0.5) / (total_adam_epochs * 0.5))
            lambda_bc = 5.0
        optimizer_adam.zero_grad()
        total_loss, (loss_pde, loss_data, loss_bc_sea, loss_bc_other, loss_reg, loss_c) = compute_loss(
            model, X_pde, X_bc, bc_labels, X_data, y_data, lambda_pde, lambda_bc
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step()
        if (epoch + 1) % 1000 == 0:
            current_lr = scheduler.get_last_lr()[0]
            with torch.no_grad():
                _, c_test = model(X_pde[:100])
            tqdm.write(
                f"Epoch {epoch + 1:5d}/{total_adam_epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Total Loss: {total_loss.item():.6f} | "
                f"PDE: {loss_pde:.6f} | Data: {loss_data:.6f} | "
                f"BC_sea: {loss_bc_sea:.6f} | BC_other: {loss_bc_other:.6f} | "
                f"Reg: {loss_reg:.6f} | C_Loss: {loss_c:.6f} | "
                f"声速范围: {c_test.min().item():.2f} ~ {c_test.max().item():.2f} m/s"
            )
    print("\n=== RMSprop微调（稳定边界损失） ===")
    optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr=1e-5)
    for epoch in tqdm(range(5000), desc="RMSprop训练进度"):
        optimizer_rmsprop.zero_grad()
        total_loss, _ = compute_loss(
            model, X_pde, X_bc, bc_labels, X_data, y_data,
            lambda_pde=config["lambda_pde_final"],
            lambda_bc=5.0
        )
        total_loss.backward()
        optimizer_rmsprop.step()
        if (epoch + 1) % 500 == 0:
            tqdm.write(f"RMSprop Epoch {epoch + 1:4d}/5000 | Loss: {total_loss.item():.8f}")
    print("\n=== LBFGS微调（精细化边界约束） ===")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=config["lr_lbfgs"],
        max_iter=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )
    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, _ = compute_loss(
            model, X_pde, X_bc, bc_labels, X_data, y_data,
            lambda_pde=config["lambda_pde_final"],
            lambda_bc=5.0
        )
        total_loss.backward()
        return total_loss
    for epoch in tqdm(range(config["epochs_lbfgs"]), desc="LBFGS训练进度"):
        loss = optimizer_lbfgs.step(closure)
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"LBFGS Epoch {epoch + 1:3d}/{config['epochs_lbfgs']} | Loss: {loss.item():.8f}")
    if not os.path.exists(os.path.dirname(config["model_save_path"])):
        os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    torch.save(model, config["model_save_path"])
    print(f"\n✅ 模型已保存至: {config['model_save_path']}")
    return model

def invert_temp_sal(model, lon_grid, lat_grid, depth_grid, sal_prior):
    lon_norm = normalize_coord(lon_grid, config["lon_min"], config["lon_max"])
    lat_norm = normalize_coord(lat_grid, config["lat_min"], config["lat_max"])
    depth_norm = normalize_coord(depth_grid, config["depth_min"], config["depth_max"])
    time_norm = np.full((len(lon_grid), len(lat_grid), len(depth_grid)), 0.5)
    lon_mesh, lat_mesh, depth_mesh = np.meshgrid(lon_norm, lat_norm, depth_norm, indexing="ij")
    input_flat = np.column_stack([
        lon_mesh.ravel(), lat_mesh.ravel(), depth_mesh.ravel(), time_norm.ravel()
    ])
    input_tensor = torch.tensor(input_flat, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        _, c_pred = model(input_tensor)
        c_pred = c_pred.cpu().numpy().reshape(lon_mesh.shape)
    def c2t(c_val, z_val, s_val):
        def f(T):
            return mackenzie(T, s_val, z_val) - c_val
        if z_val < 100:
            x0 = 28.0
        elif z_val < 300:
            x0 = 15.0
        elif z_val < 800:
            x0 = 5.0
        else:
            x0 = 2.0
        t_pred = fsolve(f, x0=x0, maxfev=2000)[0]
        t_pred = np.clip(t_pred, 0.1, 32.0)
        c_residual = abs(mackenzie(t_pred, s_val, z_val) - c_val)
        if c_residual > 2.0:
            s_val = np.clip(s_val + 0.1 * np.sign(c_val - mackenzie(t_pred, s_val, z_val)), 33.0, 36.0)
            t_pred = fsolve(f, x0=t_pred, maxfev=1000)[0]
            t_pred = np.clip(t_pred, 0.1, 32.0)
        return t_pred, s_val
    pred_temp = np.zeros_like(c_pred)
    pred_sal = np.zeros_like(c_pred)
    z_phys_mesh = denormalize_coord(depth_mesh, config["depth_min"], config["depth_max"])
    for i in range(len(lon_grid)):
        for j in range(len(lat_grid)):
            for k in range(len(depth_grid)):
                z_val = z_phys_mesh[i, j, k]
                s_val = sal_prior[k, j, i]  # WOA盐度先验
                t_pred, s_pred = c2t(c_pred[i, j, k], z_val, s_val)
                pred_temp[i, j, k] = t_pred
                pred_sal[i, j, k] = s_pred
    print(f"反演温度范围：{pred_temp.min():.2f} ~ {pred_temp.max():.2f}℃")
    print(f"反演盐度范围：{pred_sal.min():.2f} ~ {pred_sal.max():.2f}psu")
    return pred_temp, c_pred, pred_sal

def calc_metrics(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    if len(obs) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R²": np.nan, "Bias": np.nan}
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))
    r2 = 1 - np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    bias = np.mean(pred - obs)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "Bias": bias}

def validate_with_argo(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_clean):
    temp_interp = RegularGridInterpolator(
        (lon_grid, lat_grid, depth_grid),
        pred_temp,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    sal_interp = RegularGridInterpolator(
        (lon_grid, lat_grid, depth_grid),
        pred_sal,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    argo_coords = argo_clean[["LONGITUDE", "LATITUDE", "DEPTH"]].values
    argo_pred_temp = temp_interp(argo_coords)
    argo_pred_sal = sal_interp(argo_coords)
    argo_valid = argo_clean.copy()
    argo_valid["PRED_TEMP"] = argo_pred_temp
    argo_valid["PRED_SAL"] = argo_pred_sal
    argo_valid = argo_valid.dropna(subset=["PRED_TEMP", "PRED_SAL"])
    metrics_temp = calc_metrics(argo_valid["TEMP"].values, argo_valid["PRED_TEMP"].values)
    metrics_sal = calc_metrics(argo_valid["PSAL"].values, argo_valid["PRED_SAL"].values)
    print("\n=== 真实数据验证指标 ===")
    print(
        f"温度 - RMSE: {metrics_temp['RMSE']:.2f}℃, MAE: {metrics_temp['MAE']:.2f}℃, R²: {metrics_temp['R²']:.3f}, Bias: {metrics_temp['Bias']:.2f}℃")
    print(
        f"盐度 - RMSE: {metrics_sal['RMSE']:.3f}psu, MAE: {metrics_sal['MAE']:.3f}psu, R²: {metrics_sal['R²']:.3f}, Bias: {metrics_sal['Bias']:.3f}psu")
    depth_layers = [("表层", 0, 50), ("跃层", 200, 300), ("深海", 500, 1000)]
    layer_metrics_temp = []
    layer_metrics_sal = []
    for name, z_min, z_max in depth_layers:
        layer_data = argo_valid[(argo_valid["DEPTH"] >= z_min) & (argo_valid["DEPTH"] <= z_max)]
        if len(layer_data) < 10:
            continue
        met_temp = calc_metrics(layer_data["TEMP"].values, layer_data["PRED_TEMP"].values)
        layer_metrics_temp.append({"区域": name, "数据量": len(layer_data), "温度RMSE(℃)": met_temp["RMSE"]})
        met_sal = calc_metrics(layer_data["PSAL"].values, layer_data["PRED_SAL"].values)
        layer_metrics_sal.append({"区域": name, "数据量": len(layer_data), "盐度RMSE(psu)": met_sal["RMSE"]})
    print("\n=== 分层验证指标 ===")
    layer_df_temp = pd.DataFrame(layer_metrics_temp).round(3)
    layer_df_sal = pd.DataFrame(layer_metrics_sal).round(3)
    layer_df_combined = pd.merge(layer_df_temp, layer_df_sal, on=["区域", "数据量"], how="outer")
    print(layer_df_combined)
    return argo_valid, metrics_temp, metrics_sal

def plot_numerical_consistency(argo_valid, metrics_temp, metrics_sal):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    scatter_s_val = 15
    scatter_s_res = 10
    line_width_ref = 3.0
    ax1.scatter(argo_valid["TEMP"], argo_valid["PRED_TEMP"], s=scatter_s_val, alpha=0.6, c="steelblue",
                label="Data Points")
    t_min, t_max = argo_valid["TEMP"].min(), argo_valid["TEMP"].max()
    ax1.plot([t_min, t_max], [t_min, t_max], "r--", linewidth=line_width_ref, label="1:1 line")
    ax1.set_xlabel("Argo Temp (℃)")
    ax1.set_ylabel("Predicted Temp (℃)")
    ax1.set_title(f"Temp Consistency\nRMSE={metrics_temp['RMSE']:.3f}, R²={metrics_temp['R²']:.3f}")
    ax1.grid(alpha=0.4, linewidth=1.5)
    ax1.legend(loc="upper left")
    ax2.scatter(argo_valid["PSAL"], argo_valid["PRED_SAL"], s=scatter_s_val, alpha=0.6, c="darkred",
                label="Data Points")
    s_min, s_max = argo_valid["PSAL"].min(), argo_valid["PSAL"].max()
    ax2.plot([s_min, s_max], [s_min, s_max], "r--", linewidth=line_width_ref, label="1:1 line")
    ax2.set_xlabel("Argo Salt (psu)")
    ax2.set_ylabel("Predicted Salt (psu)")
    ax2.set_title(f"Salt Consistency\nRMSE={metrics_sal['RMSE']:.3f}, R²={metrics_sal['R²']:.3f}")
    ax2.grid(alpha=0.4, linewidth=1.5)
    ax2.legend(loc="upper left")
    diff_t = argo_valid["PRED_TEMP"] - argo_valid["TEMP"]
    ax3.scatter(diff_t, argo_valid["DEPTH"], s=scatter_s_res, alpha=0.5, c="steelblue")
    ax3.axvline(0, color='r', linestyle='--', linewidth=line_width_ref)
    ax3.invert_yaxis()
    ax3.set_xlabel("Temp Residual (℃)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title(f"Temp Residual Dist (Bias={metrics_temp['Bias']:.3f})")
    ax3.grid(alpha=0.4, linewidth=1.5)
    diff_s = argo_valid["PRED_SAL"] - argo_valid["PSAL"]
    ax4.scatter(diff_s, argo_valid["DEPTH"], s=scatter_s_res, alpha=0.5, c="darkred")
    ax4.axvline(0, color='r', linestyle='--', linewidth=line_width_ref)
    ax4.invert_yaxis()
    ax4.set_xlabel("Salt Residual (psu)")
    ax4.set_ylabel("Depth (m)")
    ax4.set_title(f"Salt Residual Dist (Bias={metrics_sal['Bias']:.3f})")
    ax4.grid(alpha=0.4, linewidth=1.5)
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Numerical_Consistency.png", dpi=300)
    plt.close()

def plot_spatial_distribution(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid, woa_sal=None):
    fixed_ranges = {
        "surf_temp": (22.0, 30.0),
        "surf_sal": (32.5, 34.8),
        "mid_temp": (8.0, 15.0),
        "mid_sal": (34.2, 34.8)
    }
    idx_surf = np.argmin(np.abs(depth_grid - 5))
    idx_mid = np.argmin(np.abs(depth_grid - 300))
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={"projection": ccrs.PlateCarree()})
    (ax1, ax2), (ax3, ax4) = axes
    def plot_field(ax, field_slice, argo_subset, col, cmap, title, label, v_range=None):
        if v_range is not None:
            v_min, v_max = v_range
        else:
            v_min = field_slice.min()
            v_max = field_slice.max()
            if not argo_subset.empty:
                obs_min = argo_subset[col].min()
                obs_max = argo_subset[col].max()
                v_min = min(v_min, obs_min)
                v_max = max(v_max, obs_max)
        levels = np.linspace(v_min, v_max, 25)
        cf = ax.contourf(lon_grid, lat_grid, field_slice.T, levels=levels, cmap=cmap,
                         transform=ccrs.PlateCarree(), extend='both')
        if not argo_subset.empty:
            sc = ax.scatter(argo_subset["LONGITUDE"], argo_subset["LATITUDE"],
                            c=argo_subset[col], s=80, cmap=cmap, edgecolors='k', linewidth=1.5,
                            transform=ccrs.PlateCarree(), vmin=v_min, vmax=v_max, zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
        gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=1.5, linestyle=':')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        full_title = f"{title}\nFixed Range: [{v_min:.1f}, {v_max:.1f}]"
        ax.set_title(full_title, fontsize=18, pad=10)
        cbar = plt.colorbar(cf, ax=ax, shrink=0.7, format="%.2f")  # 保留两位小数
        cbar.set_label(label, size=16)
        cbar.ax.tick_params(labelsize=14)
    argo_surf = argo_valid[argo_valid["DEPTH"] <= 10]
    argo_mid = argo_valid[(argo_valid["DEPTH"] > 250) & (argo_valid["DEPTH"] < 350)]
    plot_field(ax1, pred_temp[:, :, idx_surf], argo_surf, "TEMP", "coolwarm",
               f"Surface Temp (5m)", "Temp (℃)", v_range=fixed_ranges["surf_temp"])
    plot_field(ax2, pred_sal[:, :, idx_surf], argo_surf, "PSAL", "viridis",
               f"Surface Salt (5m)", "Salt (psu)", v_range=fixed_ranges["surf_sal"])
    plot_field(ax3, pred_temp[:, :, idx_mid], argo_mid, "TEMP", "coolwarm",
               f"Mid-Layer Temp (300m)", "Temp (℃)", v_range=fixed_ranges["mid_temp"])
    plot_field(ax4, pred_sal[:, :, idx_mid], argo_mid, "PSAL", "viridis",
               f"Mid-Layer Salt (300m)", "Salt (psu)", v_range=fixed_ranges["mid_sal"])
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Spatial_Distribution_Fixed.png", dpi=300)
    plt.close()

def plot_vertical_profile(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid):
    lon_c, lat_c = np.mean(lon_grid), np.mean(lat_grid)
    dist = np.sqrt((argo_valid["LONGITUDE"] - lon_c) ** 2 + (argo_valid["LATITUDE"] - lat_c) ** 2)
    min_idx = dist.idxmin()
    nearest_lon = argo_valid.loc[min_idx, "LONGITUDE"]
    nearest_lat = argo_valid.loc[min_idx, "LATITUDE"]
    radius = 0.2
    argo_profile = argo_valid[
        (np.abs(argo_valid["LONGITUDE"] - nearest_lon) < radius) &
        (np.abs(argo_valid["LATITUDE"] - nearest_lat) < radius)
        ].sort_values("DEPTH")
    lon_idx = np.argmin(np.abs(lon_grid - nearest_lon))
    lat_idx = np.argmin(np.abs(lat_grid - nearest_lat))
    prof_t_pred = pred_temp[lon_idx, lat_idx, :]
    prof_s_pred = pred_sal[lon_idx, lat_idx, :]
    fig, ax1 = plt.subplots(figsize=(12, 14))
    color_t = 'tab:blue'
    ax1.set_xlabel('Temperature (°C)', color=color_t, fontsize=20, fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontsize=20, fontweight='bold')
    line1, = ax1.plot(prof_t_pred, depth_grid, color=color_t, linewidth=4.0, label='PINN Pred Temp')
    scatter1 = ax1.scatter(argo_profile["TEMP"], argo_profile["DEPTH"],
                           color=color_t, marker='o', facecolors='white', edgecolors=color_t, s=100, linewidth=2.0,
                           label='Argo Obs Temp', zorder=5)
    ax1.tick_params(axis='x', labelcolor=color_t, labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.invert_yaxis()
    ax1.grid(alpha=0.4, linewidth=1.5)
    ax2 = ax1.twiny()
    color_s = 'tab:red'
    ax2.set_xlabel('Salinity (psu)', color=color_s, fontsize=20, fontweight='bold')
    line2, = ax2.plot(prof_s_pred, depth_grid, color=color_s, linewidth=4.0, linestyle='--',
                      label='PINN Pred Salt')
    scatter2 = ax2.scatter(argo_profile["PSAL"], argo_profile["DEPTH"],
                           color=color_s, marker='x', s=100, linewidth=2.5, label='Argo Obs Salt', zorder=5)
    ax2.tick_params(axis='x', labelcolor=color_s, labelsize=16)
    lines = [line1, scatter1, line2, scatter2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=16, frameon=True, shadow=True, framealpha=0.9)
    plt.title(f"Vertical Profile Comparison\nLocation: {nearest_lon:.2f}°E, {nearest_lat:.2f}°N",
              fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Vertical_Profile_With_GT.png", dpi=300)
    plt.close()

def plot_layered_error_optimized(argo_valid, temp_threshold=1.0, sal_threshold=0.3):
    bins = np.arange(0, 1100, 100)
    labels = [f"{i}-{i + 100}m" for i in range(0, 1000, 100)]
    argo_valid["depth_bin"] = pd.cut(argo_valid["DEPTH"], bins=bins, labels=labels)
    stats = []
    for lbl in labels:
        subset = argo_valid[argo_valid["depth_bin"] == lbl]
        if len(subset) > 0:
            rmse_t = np.sqrt(np.mean((subset["TEMP"] - subset["PRED_TEMP"]) ** 2))
            rmse_s = np.sqrt(np.mean((subset["PSAL"] - subset["PRED_SAL"]) ** 2))
            stats.append({"Bin": lbl, "RMSE_T": rmse_t, "RMSE_S": rmse_s})
        else:
            stats.append({"Bin": lbl, "RMSE_T": 0, "RMSE_S": 0})
    df_stats = pd.DataFrame(stats)
    fig, ax1 = plt.subplots(figsize=(16, 9))
    x = np.arange(len(labels))
    width = 0.35
    rects1 = ax1.bar(x - width / 2, df_stats["RMSE_T"], width, label='Temp RMSE', color='skyblue', edgecolor='k',
                     linewidth=1.5)
    ax1.set_ylabel('Temp RMSE (℃)', color='tab:blue', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, fontsize=14)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, df_stats["RMSE_S"], width, label='Salt RMSE', color='lightgreen', edgecolor='k',
                     linewidth=1.5)
    ax2.set_ylabel('Salt RMSE (psu)', color='tab:green', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=16)
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, rotation=90, fontweight='bold')
    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    plt.title("Layered RMSE Statistics (Precision: .3f)", fontsize=22, pad=15)
    lines, labels_l = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_l + labels2, loc="upper right", bbox_to_anchor=(0.95, 0.95), fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Layered_RMSE.png", dpi=300)
    plt.close()

def save_metrics_to_csv(metrics_temp, metrics_sal, save_dir):
    data = {
        "Metric": ["RMSE", "MAE", "R2", "Bias"],
        "Temperature": [
            metrics_temp["RMSE"], metrics_temp["MAE"], metrics_temp["R²"], metrics_temp["Bias"]
        ],
        "Salinity": [
            metrics_sal["RMSE"], metrics_sal["MAE"], metrics_sal["R²"], metrics_sal["Bias"]
        ]
    }
    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir, "evaluation_metrics.csv")
    df.to_csv(save_path, index=False, float_format="%.4f")
    print(f"\n✅ 评估指标已保存至: {save_path}")
    print(df)

if __name__ == "__main__":
    print("=== 步骤1/7：数据准备 ===")
    woa_temp, woa_sal, lon_grid, lat_grid, depth_grid = load_woa_data()
    argo_clean = load_argo_data()
    print("\n=== 步骤2/7：采样点设计 ===")
    X_pde = generate_pde_points(woa_temp, woa_sal, lon_grid, lat_grid, depth_grid)
    X_bc, bc_labels = generate_bc_points()
    X_data, y_data, p_true = generate_data_points(argo_clean, woa_temp, woa_sal, lon_grid, lat_grid, depth_grid)
    print(f"采样点数量：内部点{len(X_pde)}, 边界点{len(X_bc)}, 数据点{len(X_data)}")
    print("\n=== 采样点范围检查 ===")
    print(f"内部点（X_pde）范围：{X_pde.min().item():.3f} ~ {X_pde.max().item():.3f}")
    print(f"边界点（X_bc）范围：{X_bc.min().item():.3f} ~ {X_bc.max().item():.3f}")
    print(f"数据点（X_data）范围：{X_data.min().item():.3f} ~ {X_data.max().item():.3f}")
    print("\n=== 步骤3/7：模型初始化 ===")
    model = PINN(input_dim=4).to(device)
    print(f"模型结构：输入4维→{config['hidden_layers']}×{config['hidden_dim']}→输出2维（p,c）")
    print("\n=== 检查声速初始预测范围 ===")
    with torch.no_grad():
        _, c_test = model(X_pde[:100])
    print("初始声速c的范围：", c_test.min().item(), "~", c_test.max().item())
    with torch.no_grad():
        p_bc_init, _ = model(X_bc)
        bc_mask_0 = (bc_labels == 1) | (bc_labels == 2)
        bc_loss_init = torch.mean(p_bc_init[bc_mask_0] ** 2)
        print(f"边界点初始BC Loss（p=0约束）：{bc_loss_init.item():.2f}")
    print("\n=== 步骤4/7：模型训练 ===")
    model = train_model(model, X_pde, X_bc, bc_labels, X_data, y_data)
    print("\n=== 步骤5/7：温盐场反演 ===")
    pred_temp, pred_c, pred_sal = invert_temp_sal(model, lon_grid, lat_grid, depth_grid, woa_sal)
    print(f"反演完成：")
    print(f"  温度场形状：{pred_temp.shape}（经度×纬度×深度）")
    print(f"  盐度场形状：{pred_sal.shape}（经度×纬度×深度）")
    print(f"  声速场形状：{pred_c.shape}（经度×纬度×深度）")
    print("\n=== 步骤6/7：真实数据验证 ===")
    argo_valid, metrics_temp, metrics_sal = validate_with_argo(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_clean)
    print("\n=== 步骤7/7：结果可视化 ===")
    save_metrics_to_csv(metrics_temp, metrics_sal, config["fig_save_dir"])
    plot_numerical_consistency(argo_valid, metrics_temp, metrics_sal)
    plot_spatial_distribution(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid)
    plot_vertical_profile(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid)
    plot_layered_error_optimized(argo_valid)
    print(f"所有图表已保存至：{config['fig_save_dir']}")
    print("\n=== 全流程完成 ===")