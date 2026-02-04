import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["font.family"] = ["Microsoft YaHei", "sans-serif"]
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
print(f"U-Net (High Precision) 使用设备: {device}")
np.random.seed(42)
torch.manual_seed(42)

config = {
    "lon_min": 112.0, "lon_max": 118.0,
    "lat_min": 12.0, "lat_max": 18.0,
    "depth_min": 0.0, "depth_max": 1000.0,
    "lon_grid_num": 50, "lat_grid_num": 50, "depth_grid_num": 30,
    "epochs": 3000,
    "lr": 5e-4,
    "lambda_data": 100.0,
    "lambda_woa": 1.0,
    "lambda_tv": 0.05,
    "woa_temp_path": " ",
    "woa_sal_path": " ",
    "argo_local_path": " ",
    "fig_save_dir": " "
}
os.makedirs(config["fig_save_dir"], exist_ok=True)

class DataScaler:
    def __init__(self):
        self.t_min = 0.0
        self.t_max = 35.0
        self.s_min = 32.0
        self.s_max = 36.0

    def normalize_temp(self, t):
        return (t - self.t_min) / (self.t_max - self.t_min)

    def denormalize_temp(self, t_norm):
        return t_norm * (self.t_max - self.t_min) + self.t_min

    def normalize_sal(self, s):
        return (s - self.s_min) / (self.s_max - self.s_min)

    def denormalize_sal(self, s_norm):
        return s_norm * (self.s_max - self.s_min) + self.s_min
scaler = DataScaler()

def load_woa_data():
    print("正在读取 WOA 数据...")
    ds_temp = xr.open_dataset(config["woa_temp_path"])["t_an"]
    ds_sal = xr.open_dataset(config["woa_sal_path"])["s_an"]
    lon_grid = np.linspace(config["lon_min"], config["lon_max"], config["lon_grid_num"])
    lat_grid = np.linspace(config["lat_min"], config["lat_max"], config["lat_grid_num"])
    depth_grid = np.linspace(config["depth_min"], config["depth_max"], config["depth_grid_num"])
    woa_temp = ds_temp.sel(
        lon=slice(config["lon_min"], config["lon_max"]),
        lat=slice(config["lat_min"], config["lat_max"]),
        depth=slice(config["depth_min"], config["depth_max"])
    ).interp(lon=lon_grid, lat=lat_grid, depth=depth_grid).values
    woa_sal = ds_sal.sel(
        lon=slice(config["lon_min"], config["lon_max"]),
        lat=slice(config["lat_min"], config["lat_max"]),
        depth=slice(config["depth_min"], config["depth_max"])
    ).interp(lon=lon_grid, lat=lat_grid, depth=depth_grid).values
    for data in [woa_temp, woa_sal]:
        for k in range(data.shape[0]):
            layer = data[k, :, :]
            if np.isnan(layer).any():
                mask = np.isnan(layer)
                layer[mask] = np.nanmean(layer)
                data[k, :, :] = median_filter(layer, size=3)
    return woa_temp, woa_sal, lon_grid, lat_grid, depth_grid

def load_argo_data():
    print("正在读取 Argo 数据...")
    dfs = []
    for i in range(1, 4):
        f = os.path.join(config["argo_local_path"], f"BOA_Argo_2023_0{i}.nc")
        if os.path.exists(f):
            try:
                ds = xr.open_dataset(f, decode_times=False)
                df = ds[["lon", "lat", "pres", "temp", "salt"]].to_dataframe().reset_index()
                dfs.append(df)
            except:
                pass
    if not dfs: raise ValueError("无Argo数据文件")
    argo_df = pd.concat(dfs, ignore_index=True)
    argo_df = argo_df[
        (argo_df["lon"] >= config["lon_min"]) & (argo_df["lon"] <= config["lon_max"]) &
        (argo_df["lat"] >= config["lat_min"]) & (argo_df["lat"] <= config["lat_max"]) &
        (argo_df["pres"] >= config["depth_min"]) & (argo_df["pres"] <= config["depth_max"])
        ].dropna(subset=["temp", "salt"]).rename(columns={"pres": "depth"})
    return argo_df

def grid_argo_data(argo_df, lon_grid, lat_grid, depth_grid):
    D, H, W = len(depth_grid), len(lat_grid), len(lon_grid)
    argo_grid_temp = np.full((D, H, W), np.nan)
    argo_grid_sal = np.full((D, H, W), np.nan)
    mask = np.zeros((D, H, W))
    d_idx = np.abs(depth_grid[:, None] - argo_df["depth"].values).argmin(axis=0)
    h_idx = np.abs(lat_grid[:, None] - argo_df["lat"].values).argmin(axis=0)
    w_idx = np.abs(lon_grid[:, None] - argo_df["lon"].values).argmin(axis=0)
    argo_grid_temp[d_idx, h_idx, w_idx] = argo_df["temp"].values
    argo_grid_sal[d_idx, h_idx, w_idx] = argo_df["salt"].values
    mask[d_idx, h_idx, w_idx] = 1.0
    return argo_grid_temp, argo_grid_sal, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),  # 使用 LeakyReLU 防止梯度消失
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=5, out_channels=2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        woa_bg = x[:, 0:2, :, :, :]
        orig_size = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)
        delta = self.outc(x)
        return woa_bg + delta

def train_unet():
    woa_t, woa_s, lon_g, lat_g, dep_g = load_woa_data()
    argo_df = load_argo_data()
    argo_t_grid, argo_s_grid, mask_grid = grid_argo_data(argo_df, lon_g, lat_g, dep_g)
    woa_t_norm = scaler.normalize_temp(woa_t)
    woa_s_norm = scaler.normalize_sal(woa_s)
    input_argo_t = np.where(mask_grid == 1, argo_t_grid, woa_t)
    input_argo_s = np.where(mask_grid == 1, argo_s_grid, woa_s)
    input_argo_t_norm = scaler.normalize_temp(input_argo_t)
    input_argo_s_norm = scaler.normalize_sal(input_argo_s)
    target_t_norm = scaler.normalize_temp(argo_t_grid)
    target_s_norm = scaler.normalize_sal(argo_s_grid)
    input_data = np.stack([
        woa_t_norm, woa_s_norm, input_argo_t_norm, input_argo_s_norm, mask_grid
    ], axis=0)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    target_argo_data = np.nan_to_num(np.stack([target_t_norm, target_s_norm], axis=0))
    target_argo_tensor = torch.tensor(target_argo_data, dtype=torch.float32).unsqueeze(0).to(device)
    target_woa_tensor = torch.tensor(np.stack([woa_t_norm, woa_s_norm], axis=0), dtype=torch.float32).unsqueeze(0).to(
        device)
    mask_tensor = torch.tensor(mask_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    mask_expanded = mask_tensor.repeat(1, 2, 1, 1, 1)  # [1,2,D,H,W]
    model = ResidualUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)
    print("\n=== 开始高精度 U-Net 训练 (Residual Mode) ===")
    model.train()
    pbar = tqdm(range(config["epochs"]))
    for epoch in pbar:
        optimizer.zero_grad()
        pred = model(input_tensor)
        diff = (pred - target_argo_tensor) * mask_expanded
        loss_data = torch.sum(diff ** 2) / (torch.sum(mask_expanded) + 1e-6)  # MSE
        loss_woa = F.mse_loss(pred, target_woa_tensor)
        loss_tv = torch.mean(torch.abs(pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:])) + \
                  torch.mean(torch.abs(pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :]))
        total_loss = config["lambda_data"] * loss_data + \
                     config["lambda_woa"] * loss_woa + \
                     config["lambda_tv"] * loss_tv
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        if epoch % 100 == 0:
            pbar.set_description(
                f"Total: {total_loss.item():.4f} | Data: {loss_data.item():.5f} | WOA: {loss_woa.item():.5f}")
    return model, argo_df, lon_g, lat_g, dep_g

def calc_metrics_kan(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    if len(obs) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R²": np.nan, "Bias": np.nan}
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))
    r2 = 1 - np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    bias = np.mean(pred - obs)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "Bias": bias}

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

def plot_spatial_distribution(pred_temp, pred_sal, lon_grid, lat_grid, depth_grid, argo_valid):
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
        cbar = plt.colorbar(cf, ax=ax, shrink=0.7, format="%.2f")
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
    if dist.empty:
        print("无有效Argo数据，跳过垂直剖面绘制")
        return
    min_idx = dist.idxmin()
    nearest_lon = argo_valid.loc[min_idx, "LONGITUDE"]
    nearest_lat = argo_valid.loc[min_idx, "LATITUDE"]
    radius = 0.25
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
    line1, = ax1.plot(prof_t_pred, depth_grid, color=color_t, linewidth=4.0, label='U-Net Pred Temp')
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
    line2, = ax2.plot(prof_s_pred, depth_grid, color=color_s, linewidth=4.0, linestyle='--', label='U-Net Pred Salt')
    scatter2 = ax2.scatter(argo_profile["PSAL"], argo_profile["DEPTH"],
                           color=color_s, marker='x', s=100, linewidth=2.5, label='Argo Obs Salt', zorder=5)
    ax2.tick_params(axis='x', labelcolor=color_s, labelsize=16)
    lines = [line1, scatter1, line2, scatter2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=16, frameon=True, shadow=True, framealpha=0.9)
    plt.title(f"Vertical Profile Comparison\nLocation: {nearest_lon:.2f}°E, {nearest_lat:.2f}°N", fontsize=22, pad=20)
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

def full_evaluation(model, argo_df, lon_g, lat_g, dep_g):
    model.eval()
    print("\n=== 开始全场反演与评估 ===")
    woa_t, woa_s, _, _, _ = load_woa_data()
    _, _, mask_grid = grid_argo_data(argo_df, lon_g, lat_g, dep_g)
    argo_t_grid, argo_s_grid, _ = grid_argo_data(argo_df, lon_g, lat_g, dep_g)
    woa_t_norm = scaler.normalize_temp(woa_t)
    woa_s_norm = scaler.normalize_sal(woa_s)
    input_argo_t = np.where(mask_grid == 1, argo_t_grid, woa_t)
    input_argo_s = np.where(mask_grid == 1, argo_s_grid, woa_s)
    input_data = np.stack([
        woa_t_norm, woa_s_norm,
        scaler.normalize_temp(input_argo_t),
        scaler.normalize_sal(input_argo_s),
        mask_grid
    ], axis=0)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy()[0]
    pred_temp_3d = scaler.denormalize_temp(pred_norm[0])
    pred_sal_3d = scaler.denormalize_sal(pred_norm[1])
    pred_temp_vis = pred_temp_3d.transpose(2, 1, 0)
    pred_sal_vis = pred_sal_3d.transpose(2, 1, 0)
    interp_t = RegularGridInterpolator((dep_g, lat_g, lon_g), pred_temp_3d, bounds_error=False, fill_value=np.nan)
    interp_s = RegularGridInterpolator((dep_g, lat_g, lon_g), pred_sal_3d, bounds_error=False, fill_value=np.nan)
    points = argo_df[["depth", "lat", "lon"]].values
    argo_df["pred_temp"] = interp_t(points)
    argo_df["pred_sal"] = interp_s(points)
    argo_valid = argo_df.dropna(subset=["pred_temp", "pred_sal"]).copy()
    argo_valid = argo_valid.rename(columns={
        "lon": "LONGITUDE", "lat": "LATITUDE", "depth": "DEPTH",
        "temp": "TEMP", "salt": "PSAL",
        "pred_temp": "PRED_TEMP", "pred_sal": "PRED_SAL"
    })
    metrics_temp = calc_metrics_kan(argo_valid["TEMP"].values, argo_valid["PRED_TEMP"].values)
    metrics_sal = calc_metrics_kan(argo_valid["PSAL"].values, argo_valid["PRED_SAL"].values)
    print("\n=== 最终验证指标 ===")
    print(f"温度 RMSE: {metrics_temp['RMSE']:.4f}, R2: {metrics_temp['R²']:.4f}")
    print(f"盐度 RMSE: {metrics_sal['RMSE']:.4f}, R2: {metrics_sal['R²']:.4f}")
    save_metrics_to_csv(metrics_temp, metrics_sal, config["fig_save_dir"])
    plot_numerical_consistency(argo_valid, metrics_temp, metrics_sal)
    plot_spatial_distribution(pred_temp_vis, pred_sal_vis, lon_g, lat_g, dep_g, argo_valid)
    plot_vertical_profile(pred_temp_vis, pred_sal_vis, lon_g, lat_g, dep_g, argo_valid)
    plot_layered_error_optimized(argo_valid)
    print(f"所有图表已保存至: {config['fig_save_dir']}")

if __name__ == "__main__":
    model, argo_df, lon_g, lat_g, dep_g = train_unet()
    full_evaluation(model, argo_df, lon_g, lat_g, dep_g)