import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import torch.fft
import torch.cuda.amp as amp
import math
from scipy.ndimage import median_filter

plt.rcParams["font.family"] = ["Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
np.random.seed(66)
torch.manual_seed(66)

config = {
    "lon_min": 112.0, "lon_max": 118.0,
    "lat_min": 12.0, "lat_max": 18.0,
    "depth_min": 0.0, "depth_max": 1000.0,
    "time_min": 2023.50,
    "time_max": 2023.75,
    "lon_grid_num": 64, "lat_grid_num": 64, "depth_grid_num": 64,
    "N_pde": 8000, "N_pde_thermo": 5000,
    "N_pde_surface": 2500,
    "N_bc": 5000,
    "temp_min": 0.0, "temp_max": 32.0,
    "sal_min": 28.0, "sal_max": 37.0,
    "fno_modes": 12,
    "fno_width": 64,
    "lr_adam": 2e-3, "lr_lbfgs": 1.0,
    "epochs_adam": 4000, "epochs_lbfgs": 0,
    "lambda_pde_init": 0.01, "lambda_pde_final": 2.0, "lambda_data": 200.0,
    "lambda_bc": 10.0, "lambda_temp": 1.0,
    "lambda_sal": 1.0,
    "lambda_reg": 1e-3,
    "woa_temp_path": " ",
    "woa_sal_path": " ",
    "argo_local_path": " ",
    "model_save_path": " ",
    "fig_save_dir": " "
}
os.makedirs(config["fig_save_dir"], exist_ok=True)

def load_woa_data():
    print(f"正在读取 WOA 数据: {config['woa_temp_path']}")
    try:
        ds_temp = xr.open_dataset(config["woa_temp_path"], decode_times=False)
        if "t_an" in ds_temp:
            ds_temp = ds_temp["t_an"]
        elif "t_mn" in ds_temp:
            ds_temp = ds_temp["t_mn"]
        else:
            var_name = list(ds_temp.data_vars)[0]
            print(f"警告：未找到标准温度变量名，使用 '{var_name}'")
            ds_temp = ds_temp[var_name]
        if "time" in ds_temp.dims:
            ds_temp = ds_temp.isel(time=0)
        ds_sal = xr.open_dataset(config["woa_sal_path"], decode_times=False)
        if "s_an" in ds_sal:
            ds_sal = ds_sal["s_an"]
        elif "s_mn" in ds_sal:
            ds_sal = ds_sal["s_mn"]
        else:
            var_name = list(ds_sal.data_vars)[0]
            print(f"警告：未找到标准盐度变量名，使用 '{var_name}'")
            ds_sal = ds_sal[var_name]

        if "time" in ds_sal.dims:
            ds_sal = ds_sal.isel(time=0)
    except Exception as e:
        print(f"读取 WOA 文件失败，请检查路径或文件格式。错误信息: {e}")
        raise e
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
            layer = median_filter(layer, size=3, mode='nearest')  # 若点太少，用中值滤波兜底
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
    c_true_woa = mackenzie(woa_temp_interp, woa_sal_interp, z_mesh)
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
        time_start = datetime(2023, 7, 1)  # 修改：2009 -> 2014
        time_end = datetime(2023, 9, 30)  # 修改：2009 -> 2014
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
        (~argo_df["salt_scatter_error"].isna())
        ].copy()
    argo_clean = argo_clean.dropna(subset=["lon", "lat", "pres", "temp", "salt"])
    if len(argo_clean) == 0:
        print("警告：严格误差阈值后无数据，尝试放宽条件...")
        argo_clean = argo_df[
            (~argo_df["temp_scatter_error"].isna()) &
            (~argo_df["salt_scatter_error"].isna()) &
            (argo_df["temp_scatter_error"] < 5.0) &  # 极大放宽
            (argo_df["salt_scatter_error"] < 5.0)
            ].copy()
        argo_clean = argo_clean.dropna(subset=["lon", "lat", "pres", "temp", "salt"])
    if len(argo_clean) == 0:
        print("警告：无法读取误差列(可能是NaN)，正在忽略误差限制，仅保留温盐有效的记录...")
        argo_clean = argo_df.dropna(subset=["lon", "lat", "pres", "temp", "salt"]).copy()
    if len(argo_clean) == 0:
        print("调试信息 - 数据列缺失情况：")
        print(f"Total rows: {len(argo_df)}")
        print(f"NaN in Temp: {argo_df['temp'].isna().sum()}")
        print(f"NaN in Salt: {argo_df['salt'].isna().sum()}")
        raise ValueError("所有数据均为无效值（Temp/Salt 含NaN），请检查原始数据完整性")
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
    pde_norm = np.vstack([pde_norm, thermo_norm, surface_norm])
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
        np.ones(N_sea),
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
    argo_valid = argo_clean.dropna(subset=["LONGITUDE", "LATITUDE", "DEPTH", "TEMP", "PSAL"])
    lon_norm = normalize_coord(argo_valid["LONGITUDE"].values, config["lon_min"], config["lon_max"])
    lat_norm = normalize_coord(argo_valid["LATITUDE"].values, config["lat_min"], config["lat_max"])
    depth_norm = normalize_coord(argo_valid["DEPTH"].values, config["depth_min"], config["depth_max"])
    time_norm = np.random.uniform(0, 1, size=len(argo_valid))
    X_data_np = np.column_stack([lon_norm, lat_norm, depth_norm, time_norm])
    X_data = torch.tensor(X_data_np, dtype=torch.float32).to(device)
    temp_phys = argo_valid["TEMP"].values
    sal_phys = argo_valid["PSAL"].values
    temp_target = (temp_phys - config["temp_min"]) / (config["temp_max"] - config["temp_min"])
    sal_target = (sal_phys - config["sal_min"]) / (config["sal_max"] - config["sal_min"])
    y_target_np = np.column_stack([temp_target, sal_target])
    y_data = torch.tensor(y_target_np, dtype=torch.float32).to(device)
    print(f"训练数据构建完成: {len(X_data)} 点")
    print(f"Temp 归一化范围: {y_data[:, 0].min():.3f} ~ {y_data[:, 0].max():.3f}")
    print(f"Salt 归一化范围: {y_data[:, 1].min():.3f} ~ {y_data[:, 1].max():.3f}")
    return X_data, y_data

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Depth
        self.modes2 = modes2  # Lat
        self.modes3 = modes3  # Lon
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2, dtype=torch.float32))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2, dtype=torch.float32))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2, dtype=torch.float32))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                                 dtype=torch.cfloat, device=x.device)
            w1 = torch.view_as_complex(self.weights1)
            w2 = torch.view_as_complex(self.weights2)
            w3 = torch.view_as_complex(self.weights3)
            w4 = torch.view_as_complex(self.weights4)
            out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], w1)
            out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], w2)
            out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], w3)
            out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], w4)
            x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline * noise).permute(2, 1, 0)
            )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)

class PI_FNO(nn.Module):
    def __init__(self, modes=12, width=32):
        super(PI_FNO, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.modes3 = modes
        self.width = width
        self.fc0 = nn.Linear(6, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.kan1 = KANLinear(self.width, 128, grid_size=10, spline_order=3)
        self.kan2 = KANLinear(128, 2, grid_size=10, spline_order=3)

    def forward(self, x):
        woa_background = x[..., 4:].clone()  # (batch, d, lat, lon, 2)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.kan1(x)
        correction = self.kan2(x)
        out = woa_background + correction
        out = torch.clamp(out, 0.0, 1.0)
        return out

def laplacian_3d_fdm(field, dx, dy, dz):
    d2_dz2 = (field[:, 2:, :, :, :] - 2 * field[:, 1:-1, :, :, :] + field[:, :-2, :, :, :]) / (dz ** 2)
    d2_dy2 = (field[:, :, 2:, :, :] - 2 * field[:, :, 1:-1, :, :] + field[:, :, :-2, :, :]) / (dy ** 2)
    d2_dx2 = (field[:, :, :, 2:, :] - 2 * field[:, :, :, 1:-1, :] + field[:, :, :, :-2, :]) / (dx ** 2)
    core_z = d2_dz2[:, :, 1:-1, 1:-1, :]
    core_y = d2_dy2[:, 1:-1, :, 1:-1, :]
    core_x = d2_dx2[:, 1:-1, 1:-1, :, :]
    laplacian = core_x + core_y + core_z
    return laplacian, field[:, 1:-1, 1:-1, 1:-1, :]

def compute_loss_fno(model, grid_input, X_data, y_data):
    out_full = model(grid_input)  # (Batch, Depth, Lat, Lon, 2)
    sample_coords = X_data[:, :3].clone()
    sample_coords = sample_coords * 2.0 - 1.0
    grid_sample_coords = sample_coords.view(1, 1, 1, -1, 3)
    out_permuted = out_full.permute(0, 4, 1, 2, 3)
    pred_sampled = F.grid_sample(out_permuted, grid_sample_coords, align_corners=True).view(2, -1).permute(1, 0)
    depth_vals = X_data[:, 2]
    spatial_weights = 5.0 + 100.0 * torch.exp(-50.0 * depth_vals)
    diff_temp = pred_sampled[:, 0] - y_data[:, 0]
    loss_temp = torch.mean(spatial_weights * (diff_temp ** 2))
    diff_sal = pred_sampled[:, 1] - y_data[:, 1]
    loss_sal = torch.mean(spatial_weights * (diff_sal ** 2))
    z_coords = torch.linspace(0, 1, config["depth_grid_num"], device=grid_input.device)
    dz = out_full[:, 1:, :, :, :] - out_full[:, :-1, :, :, :]
    z_mid = (z_coords[1:] + z_coords[:-1]) / 2.0
    slope_mask = (z_mid.view(1, -1, 1, 1, 1)) ** 2
    loss_reg_slope = torch.mean(slope_mask * (dz ** 2))
    dzz = dz[:, 1:, :, :, :] - dz[:, :-1, :, :, :]
    z_inner = z_coords[1:-1]
    curv_mask = 1.0 + 1000.0 * (z_inner.view(1, -1, 1, 1, 1) ** 4)
    loss_reg_curvature = torch.mean(curv_mask * (dzz ** 2))
    loss_reg = loss_reg_slope + loss_reg_curvature
    total_loss = config["lambda_temp"] * loss_temp + \
                 config["lambda_sal"] * loss_sal + \
                 config["lambda_reg"] * loss_reg
    return total_loss, (loss_temp.item(), loss_sal.item())

def train_model_fno(model, X_data, y_data, woa_temp_field, woa_sal_field):
    lons = np.linspace(0, 1, config["lon_grid_num"])
    lats = np.linspace(0, 1, config["lat_grid_num"])
    depths = np.linspace(0, 1, config["depth_grid_num"])
    time_val = 0.5
    depth_mesh, lat_mesh, lon_mesh = np.meshgrid(depths, lats, lons, indexing='ij')
    time_mesh = np.full_like(depth_mesh, time_val)
    woa_t_norm = (woa_temp_field - config["temp_min"]) / (config["temp_max"] - config["temp_min"])
    woa_s_norm = (woa_sal_field - config["sal_min"]) / (config["sal_max"] - config["sal_min"])
    woa_t_norm = np.clip(woa_t_norm, 0, 1)
    woa_s_norm = np.clip(woa_s_norm, 0, 1)
    grid_np = np.stack([
        lon_mesh,
        lat_mesh,
        depth_mesh,
        time_mesh,
        woa_t_norm,
        woa_s_norm
    ], axis=-1)
    grid_input = torch.tensor(grid_np, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Grid Input Shape: {grid_input.shape} (Should be 1, D, Lat, Lon, 6)")
    scaler = amp.GradScaler()
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=config["lr_adam"])
    scheduler = CosineAnnealingLR(optimizer_adam, T_max=config["epochs_adam"])
    print(f"=== 阶段 1: Adam 训练 ({config['epochs_adam']} epochs) ===")
    model.train()
    pbar = tqdm(range(config["epochs_adam"]), desc="Adam Training")
    for epoch in pbar:
        optimizer_adam.zero_grad()
        with amp.autocast():
            loss, (l_temp, l_sal) = compute_loss_fno(model, grid_input, X_data, y_data)
        scaler.scale(loss).backward()
        scaler.step(optimizer_adam)
        scaler.update()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
    optimizer_adam = None
    torch.cuda.empty_cache()
    print(f"\n=== 阶段 2: L-BFGS 微调 (Max {config['epochs_lbfgs']} Iterations) ===")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=config["lr_lbfgs"],
        max_iter=config["epochs_lbfgs"],
        max_eval=int(config["epochs_lbfgs"] * 1.25),
        history_size=50,
        tolerance_grad=1e-15,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe"
    )
    pbar_lbfgs = tqdm(total=config["epochs_lbfgs"], desc="L-BFGS Steps")
    def closure():
        pbar_lbfgs.update(1)
        optimizer_lbfgs.zero_grad()
        loss, (l_temp, l_sal) = compute_loss_fno(model, grid_input, X_data, y_data)
        loss.backward()
        pbar_lbfgs.set_postfix({"Loss": f"{loss.item():.6f}"})
        return loss
    try:
        optimizer_lbfgs.step(closure)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n警告: L-BFGS 显存不足，跳过。")
        else:
            raise e
    pbar_lbfgs.close()
    with torch.no_grad():
        final_loss, _ = compute_loss_fno(model, grid_input, X_data, y_data)
    print(f"训练结束。Final Loss: {final_loss.item():.6f}")
    if hasattr(model, 'module'):
        torch.save(model.module, config["model_save_path"])
    else:
        torch.save(model, config["model_save_path"])
    return model, grid_input

def invert_temp_sal_fno(model, grid_input):
    print("正在生成全场温盐数据...")
    model.eval()
    with torch.no_grad():
        out_full = model(grid_input)  # (1, D, Lat, Lon, 2)
        out_np = out_full.squeeze().cpu().numpy()  # (D, Lat, Lon, 2)
    t_norm = out_np[..., 0]
    s_norm = out_np[..., 1]
    pred_temp = t_norm * (config["temp_max"] - config["temp_min"]) + config["temp_min"]
    pred_sal = s_norm * (config["sal_max"] - config["sal_min"]) + config["sal_min"]
    z_vals = np.linspace(config["depth_min"], config["depth_max"], config["depth_grid_num"])
    z_mesh = np.tile(z_vals.reshape(-1, 1, 1), (1, config["lat_grid_num"], config["lon_grid_num"]))
    pred_c = mackenzie(pred_temp, pred_sal, z_mesh)
    pred_temp = pred_temp.transpose(2, 1, 0)
    pred_sal = pred_sal.transpose(2, 1, 0)
    pred_c = pred_c.transpose(2, 1, 0)
    return pred_temp, pred_c, pred_sal

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
        pred_temp, method="linear", bounds_error=False, fill_value=np.nan
    )
    sal_interp = RegularGridInterpolator(
        (lon_grid, lat_grid, depth_grid),
        pred_sal, method="linear", bounds_error=False, fill_value=np.nan
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
    print(f"温度 - RMSE: {metrics_temp['RMSE']:.3f}℃, R²: {metrics_temp['R²']:.3f}")
    print(f"盐度 - RMSE: {metrics_sal['RMSE']:.3f}psu, R²: {metrics_sal['R²']:.3f}")
    return argo_valid, metrics_temp, metrics_sal

def calc_profile_rmse(argo_valid):
    argo_valid = argo_valid.copy()
    argo_valid['profile_id'] = (
            argo_valid['TIME'].astype(str) + "_" +
            argo_valid['LONGITUDE'].round(3).astype(str) + "_" +
            argo_valid['LATITUDE'].round(3).astype(str)
    )
    profile_metrics = []
    for pid, group in argo_valid.groupby('profile_id'):
        if len(group) < 3: continue
        rmse_t = np.sqrt(np.mean((group['PRED_TEMP'] - group['TEMP']) ** 2))
        rmse_s = np.sqrt(np.mean((group['PRED_SAL'] - group['PSAL']) ** 2))
        profile_metrics.append({
            'profile_id': pid,
            'lon': group['LONGITUDE'].iloc[0],
            'lat': group['LATITUDE'].iloc[0],
            'time': group['TIME'].iloc[0],
            'RMSE_T': rmse_t,
            'RMSE_S': rmse_s
        })
    return pd.DataFrame(profile_metrics)

def find_closest_profiles(df_profiles, target_points):
    results = []
    if df_profiles.empty:
        print("警告：未计算出有效的剖面 RMSE，跳过 Table 6 对比。")
        return pd.DataFrame()
    for name, (target_lat, target_lon) in target_points.items():
        current_df = df_profiles
        dist = np.sqrt((current_df['lat'] - target_lat) ** 2 + (current_df['lon'] - target_lon) ** 2)
        if len(dist) == 0: continue
        closest_idx = dist.idxmin()
        closest_profile = current_df.loc[closest_idx]
        min_dist = dist.min()
        results.append({
            'Target_Name': name,
            'Target_Loc': f"({target_lat}, {target_lon})",
            'Found_Loc': f"({closest_profile['lat']:.3f}, {closest_profile['lon']:.3f})",
            'Time': closest_profile['time'],
            'RMSE_T': closest_profile['RMSE_T'],
            'RMSE_S': closest_profile['RMSE_S'],
            'Distance_deg': min_dist
        })
    return pd.DataFrame(results)

def plot_numerical_consistency(argo_valid, metrics_temp, metrics_sal):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams.update({"font.size": 10})
    ax1.scatter(argo_valid["TEMP"], argo_valid["PRED_TEMP"], s=2, alpha=0.6, c="steelblue")
    t_min, t_max = argo_valid["TEMP"].min(), argo_valid["TEMP"].max()
    ax1.plot([t_min, t_max], [t_min, t_max], "r--", linewidth=1.5, label="1:1 line")
    ax1.set_xlabel("Argo Temp (℃)")
    ax1.set_ylabel("Predicted Temp (℃)")
    ax1.set_title(f"Temp Consistency\nRMSE={metrics_temp['RMSE']:.3f}, R²={metrics_temp['R²']:.3f}")
    ax1.grid(alpha=0.3)
    ax2.scatter(argo_valid["PSAL"], argo_valid["PRED_SAL"], s=2, alpha=0.6, c="darkred")
    s_min, s_max = argo_valid["PSAL"].min(), argo_valid["PSAL"].max()
    ax2.plot([s_min, s_max], [s_min, s_max], "r--", linewidth=1.5)
    ax2.set_xlabel("Argo Salt (psu)")
    ax2.set_ylabel("Predicted Salt (psu)")
    ax2.set_title(f"Salt Consistency\nRMSE={metrics_sal['RMSE']:.3f}, R²={metrics_sal['R²']:.3f}")
    ax2.grid(alpha=0.3)
    diff_t = argo_valid["PRED_TEMP"] - argo_valid["TEMP"]
    ax3.scatter(diff_t, argo_valid["DEPTH"], s=1, alpha=0.5, c="steelblue")
    ax3.axvline(0, color='r', linestyle='--')
    ax3.invert_yaxis()
    ax3.set_xlabel("Temp Residual (℃)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title(f"Temp Residual Dist (Bias={metrics_temp['Bias']:.3f})")
    ax3.grid(alpha=0.3)
    diff_s = argo_valid["PRED_SAL"] - argo_valid["PSAL"]
    ax4.scatter(diff_s, argo_valid["DEPTH"], s=1, alpha=0.5, c="darkred")
    ax4.axvline(0, color='r', linestyle='--')
    ax4.invert_yaxis()
    ax4.set_xlabel("Salt Residual (psu)")
    ax4.set_ylabel("Depth (m)")
    ax4.set_title(f"Salt Residual Dist (Bias={metrics_sal['Bias']:.3f})")
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Numerical_Consistency.png", dpi=300)
    plt.close()

def plot_spatial_distribution(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid):
    idx_surf = np.argmin(np.abs(depth_grid - 5))
    idx_mid = np.argmin(np.abs(depth_grid - 300))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    (ax1, ax2), (ax3, ax4) = axes
    def plot_field(ax, field_slice, argo_subset, col, cmap, title, label):
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
                            c=argo_subset[col], s=40, cmap=cmap, edgecolors='k', linewidth=0.5,
                            transform=ccrs.PlateCarree(), vmin=v_min, vmax=v_max, zorder=10)
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(draw_labels=True, alpha=0.3)
        full_title = f"{title}\nRange: [{v_min:.3f}, {v_max:.3f}]"
        ax.set_title(full_title, fontsize=11)
        plt.colorbar(cf, ax=ax, shrink=0.7, label=label, format="%.3f")
    argo_surf = argo_valid[argo_valid["DEPTH"] <= 10]
    argo_mid = argo_valid[(argo_valid["DEPTH"] > 250) & (argo_valid["DEPTH"] < 350)]
    plot_field(ax1, pred_temp[:, :, idx_surf], argo_surf, "TEMP", "coolwarm",
               f"Surface Temp (5m)", "Temp (℃)")
    plot_field(ax2, pred_sal[:, :, idx_surf], argo_surf, "PSAL", "viridis",
               f"Surface Salt (5m)", "Salt (psu)")
    plot_field(ax3, pred_temp[:, :, idx_mid], argo_mid, "TEMP", "coolwarm",
               f"Mid-Layer Temp (300m)", "Temp (℃)")
    plot_field(ax4, pred_sal[:, :, idx_mid], argo_mid, "PSAL", "viridis",
               f"Mid-Layer Salt (300m)", "Salt (psu)")
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Spatial_Distribution.png", dpi=300)
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
    fig, ax1 = plt.subplots(figsize=(10, 12))
    color_t = 'tab:blue'
    ax1.set_xlabel('Temperature (°C)', color=color_t, fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    line1, = ax1.plot(prof_t_pred, depth_grid, color=color_t, linewidth=2.5, label='FNO Pred Temp')
    scatter1 = ax1.scatter(argo_profile["TEMP"], argo_profile["DEPTH"],
                           color=color_t, marker='o', facecolors='white', edgecolors=color_t, s=40,
                           label='Argo Obs Temp', zorder=5)
    ax1.tick_params(axis='x', labelcolor=color_t)
    ax1.invert_yaxis()
    ax1.grid(alpha=0.3)
    ax2 = ax1.twiny()
    color_s = 'tab:red'
    ax2.set_xlabel('Salinity (psu)', color=color_s, fontsize=12)
    line2, = ax2.plot(prof_s_pred, depth_grid, color=color_s, linewidth=2.5, linestyle='--', label='FNO Pred Salt')
    # Argo 真实点 (叉号)
    scatter2 = ax2.scatter(argo_profile["PSAL"], argo_profile["DEPTH"],
                           color=color_s, marker='x', s=40, label='Argo Obs Salt', zorder=5)
    ax2.tick_params(axis='x', labelcolor=color_s)
    lines = [line1, scatter1, line2, scatter2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=10, frameon=True, shadow=True)
    plt.title(f"Vertical Profile Comparison\nLocation: {nearest_lon:.2f}°E, {nearest_lat:.2f}°N", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Vertical_Profile_With_GT.png", dpi=300)
    plt.close()
    print(f"垂直剖面对比图已保存 (参考点位于 {nearest_lon:.2f}E, {nearest_lat:.2f}N)")

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
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    rects1 = ax1.bar(x - width / 2, df_stats["RMSE_T"], width, label='Temp RMSE', color='skyblue', edgecolor='k')
    ax1.set_ylabel('Temp RMSE (℃)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, df_stats["RMSE_S"], width, label='Salt RMSE', color='lightgreen', edgecolor='k')
    ax2.set_ylabel('Salt RMSE (psu)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)
    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    plt.title("Layered RMSE Statistics (Precision: .3f)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
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
    print("\n=== 正在执行基于地理位置的防泄露划分 (Location-based Hold-out Split) ===")
    print("策略：在 2023 年数据中，寻找与论文目标点位 (Point 1 & 2) 距离最近的剖面作为独立测试集。")
    print("      其余所有数据将用于模型训练。")
    paper_targets = [
        # Point 1: 13.602°N, 113.3°E
        {"name": "Point 1", "lat": 13.602, "lon": 113.300, "target_month": 7},
        # Point 2: 14.178°N, 112.296°E (目标日期：2014-07-18)
        {"name": "Point 2", "lat": 14.178, "lon": 112.296, "target_month": 7}
    ]
    test_indices = []
    found_profiles_info = []
    for target in paper_targets:
        candidates = argo_clean[argo_clean['TIME'].dt.month == target['target_month']]
        if candidates.empty:
            print(f"警告：未找到 2014年 {target['target_month']} 月份的数据用于 {target['name']}")
            continue
        dist = np.sqrt(
            (candidates['LATITUDE'] - target['lat']) ** 2 +
            (candidates['LONGITUDE'] - target['lon']) ** 2
        )
        closest_idx = dist.idxmin()
        closest_row = candidates.loc[closest_idx]
        min_dist = dist.min()
        c_time = closest_row['TIME']
        c_lat = closest_row['LATITUDE']
        c_lon = closest_row['LONGITUDE']
        found_profiles_info.append({
            "Target": target['name'],
            "Paper_Loc": f"{target['lat']}N, {target['lon']}E",
            "Found_Loc": f"{c_lat:.3f}N, {c_lon:.3f}E",
            "Distance": min_dist,
            "Time": c_time
        })
        mask = (argo_clean['TIME'] == c_time) & \
               (np.abs(argo_clean['LATITUDE'] - c_lat) < 1e-4) & \
               (np.abs(argo_clean['LONGITUDE'] - c_lon) < 1e-4)
        test_indices.extend(argo_clean[mask].index.tolist())
    print("-" * 80)
    print(f"{'Target Point':<15} | {'Paper Loc':<20} | {'Found Nearest Loc':<20} | {'Dev. Dist':<10}")
    print("-" * 80)
    for info in found_profiles_info:
        print(f"{info['Target']:<15} | {info['Paper_Loc']:<20} | {info['Found_Loc']:<20} | {info['Distance']:.4f}°")
    print("-" * 80)
    final_test_mask = argo_clean.index.isin(test_indices)
    test_set_df = argo_clean[final_test_mask].copy()  # 选中的最近剖面 (用于 Table 6 对比，模型完全不可见)
    train_set_df = argo_clean[~final_test_mask].copy()  # 剩余所有数据 (用于训练)
    print(f"\n原始数据总量: {len(argo_clean)}")
    print(f"训练集数量 (Train): {len(train_set_df)}")
    print(f"测试集数量 (Test): {len(test_set_df)} (包含 {len(paper_targets)} 个完整剖面)")
    if len(test_set_df) == 0:
        raise ValueError("严重错误：未能选出测试集，请检查 Argo 数据是否包含 2023 年 1 月的数据！")
    print("\n=== 步骤2/7：采样点设计 ===")
    X_pde = generate_pde_points(woa_temp, woa_sal, lon_grid, lat_grid, depth_grid)
    X_bc, bc_labels = generate_bc_points()
    X_data, y_data = generate_data_points(train_set_df, woa_temp, woa_sal, lon_grid, lat_grid, depth_grid)
    print(f"采样点数量：内部点{len(X_pde)}, 边界点{len(X_bc)}, 数据点{len(X_data)}")
    print("\n=== 步骤3/7：模型初始化 (PI-FNO) ===")
    model = PI_FNO(modes=config["fno_modes"], width=config["fno_width"])
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行模式！")
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    print("\n=== 步骤4/7：模型训练 (Residual Mode) ===")
    model, grid_input = train_model_fno(model, X_data, y_data, woa_temp, woa_sal)
    print("\n=== 步骤5/7：温盐场反演 (生成全场数据) ===")
    pred_temp, pred_c, pred_sal = invert_temp_sal_fno(model, grid_input)
    print("\n=== 步骤6/7：真实数据验证 (基于 Hold-out Test Set) ===")
    argo_valid_test, metrics_temp, metrics_sal = validate_with_argo(
        pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, test_set_df
    )
    save_metrics_to_csv(metrics_temp, metrics_sal, config["fig_save_dir"])
    print("\n=== 附加分析：模拟论文 Table 6 对比 (基于 Location-based Hold-out) ===")
    df_profiles = calc_profile_rmse(argo_valid_test)
    if not df_profiles.empty:
        print(f"测试集平均剖面 RMSE (Temp): {df_profiles['RMSE_T'].mean():.4f} °C")
        print(f"测试集平均剖面 RMSE (Salt): {df_profiles['RMSE_S'].mean():.4f} psu")
        paper_points_dict = {
            "Point 1 (13.6N, 113.3E)": (13.602, 113.3),
            "Point 2 (14.2N, 112.3E)": (14.178, 112.296)
        }
        comparison_table = find_closest_profiles(df_profiles, paper_points_dict)
        print("\n与论文点位最接近的 2023 年剖面误差统计 (独立测试集)：")
        print(comparison_table.to_string())
        comparison_table.to_csv(os.path.join(config["fig_save_dir"], "table6_simulation_holdout_optimized.csv"))
    else:
        print("无有效剖面数据，跳过对比。")
    print("\n=== 步骤7/7：结果可视化 (基于 Test Set) ===")
    plot_numerical_consistency(argo_valid_test, metrics_temp, metrics_sal)
    plot_spatial_distribution(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid_test)
    plot_vertical_profile(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid_test)
    plot_layered_error_optimized(argo_valid_test)
    print(f"所有验证图表已保存至：{config['fig_save_dir']}")
    print("\n=== 全流程完成 (已执行基于位置优化的防泄露验证) ===")