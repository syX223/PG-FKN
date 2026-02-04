import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pykrige.ok3d import OrdinaryKriging3D
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

config = {
    "lon_min": 112.0, "lon_max": 118.0,
    "lat_min": 12.0, "lat_max": 18.0,
    "depth_min": 0.0, "depth_max": 1000.0,
    "lon_grid_num": 64, "lat_grid_num": 64, "depth_grid_num": 64,

    "argo_local_path": " ",
    "fig_save_dir": " "
}
os.makedirs(config["fig_save_dir"], exist_ok=True)

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

def load_argo_data_for_kriging():
    argo_files = [
        os.path.join(config["argo_local_path"], " "),
        os.path.join(config["argo_local_path"], " "),
        os.path.join(config["argo_local_path"], " ")
    ]
    argo_dfs = []
    for nc_file in argo_files:
        if not os.path.exists(nc_file):
            print(f"Warning: Argo文件不存在：{nc_file}")
            continue
        try:
            ds = xr.open_dataset(nc_file, decode_times=False)
            core_vars = ["lon", "lat", "pres", "temp", "salt"]
            # 确保变量存在
            available_vars = [v for v in core_vars if v in ds.variables]
            if len(available_vars) != len(core_vars):
                continue
            df = ds[available_vars].to_dataframe().reset_index()
            df = df.rename(columns={
                "lon": "LONGITUDE", "lat": "LATITUDE",
                "pres": "DEPTH", "temp": "TEMP", "salt": "PSAL"
            })
            argo_dfs.append(df)
        except Exception as e:
            print(f"读取{nc_file}失败：{e}")
            continue
    if not argo_dfs:
        raise ValueError("未加载到任何Argo数据")
    argo_df = pd.concat(argo_dfs, ignore_index=True)
    argo_df = argo_df[
        (argo_df["LONGITUDE"] >= config["lon_min"]) & (argo_df["LONGITUDE"] <= config["lon_max"]) &
        (argo_df["LATITUDE"] >= config["lat_min"]) & (argo_df["LATITUDE"] <= config["lat_max"]) &
        (argo_df["DEPTH"] >= config["depth_min"]) & (argo_df["DEPTH"] <= config["depth_max"])
        ].dropna(subset=["LONGITUDE", "LATITUDE", "DEPTH", "TEMP", "PSAL"])
    print(f"Argo数据加载完成：{len(argo_df)}条有效数据")
    return argo_df

def run_kriging_3d(argo_df):
    print("--- 正在进行数据预处理 (去重 + Jitter) ---")
    before_len = len(argo_df)
    argo_df = argo_df.drop_duplicates(subset=["LONGITUDE", "LATITUDE", "DEPTH"])
    after_len = len(argo_df)
    if before_len - after_len > 0:
        print(f"发现并移除了 {before_len - after_len} 个重复数据点")
    jitter_strength = 1e-5 
    argo_df = argo_df.copy()
    argo_df["LONGITUDE"] += np.random.uniform(-jitter_strength, jitter_strength, size=len(argo_df))
    argo_df["LATITUDE"] += np.random.uniform(-jitter_strength, jitter_strength, size=len(argo_df))
    argo_df["DEPTH"] += np.random.uniform(-0.001, 0.001, size=len(argo_df))
    x_raw = argo_df["LONGITUDE"].values
    y_raw = argo_df["LATITUDE"].values
    z_raw = argo_df["DEPTH"].values
    temp_vals = argo_df["TEMP"].values
    sal_vals = argo_df["PSAL"].values
    x_min, x_max = config["lon_min"], config["lon_max"]
    y_min, y_max = config["lat_min"], config["lat_max"]
    z_min, z_max = config["depth_min"], config["depth_max"]
    def normalize(v, v_min, v_max):
        return (v - v_min) / (v_max - v_min)
    x_norm = normalize(x_raw, x_min, x_max)
    y_norm = normalize(y_raw, y_min, y_max)
    z_norm = normalize(z_raw, z_min, z_max)
    lon_grid = np.linspace(config["lon_min"], config["lon_max"], config["lon_grid_num"])
    lat_grid = np.linspace(config["lat_min"], config["lat_max"], config["lat_grid_num"])
    depth_grid = np.linspace(config["depth_min"], config["depth_max"], config["depth_grid_num"])
    lon_grid_norm = normalize(lon_grid, x_min, x_max)
    lat_grid_norm = normalize(lat_grid, y_min, y_max)
    depth_grid_norm = normalize(depth_grid, z_min, z_max)
    lon_mesh, lat_mesh, depth_mesh = np.meshgrid(lon_grid, lat_grid, depth_grid, indexing="ij")
    print(f"正在进行温度场 3D Kriging 插值 (Normalized, {config['lon_grid_num']}x{config['lat_grid_num']}x{config['depth_grid_num']})...")
    ok3d_temp = OrdinaryKriging3D(
        x_norm, y_norm, z_norm, temp_vals,
        variogram_model="linear", 
        nlags=10,                  
        verbose=True,
        enable_plotting=False
    )
    kriging_temp_flat, _ = ok3d_temp.execute("grid", lon_grid_norm, lat_grid_norm, depth_grid_norm)
    if kriging_temp_flat.shape == (config["depth_grid_num"], config["lat_grid_num"], config["lon_grid_num"]):
        kriging_temp = kriging_temp_flat.transpose(2, 1, 0) # (d, lat, lon) -> (lon, lat, d)
    else:
        kriging_temp = kriging_temp_flat.reshape(lon_mesh.shape)
    print("正在进行盐度场 3D Kriging 插值 (Normalized)...")
    ok3d_sal = OrdinaryKriging3D(
        x_norm, y_norm, z_norm, sal_vals,
        variogram_model="linear", 
        nlags=10,                 
        verbose=True,
        enable_plotting=False
    )
    kriging_sal_flat, _ = ok3d_sal.execute("grid", lon_grid_norm, lat_grid_norm, depth_grid_norm)
    if kriging_sal_flat.shape == (config["depth_grid_num"], config["lat_grid_num"], config["lon_grid_num"]):
        kriging_sal = kriging_sal_flat.transpose(2, 1, 0)
    else:
        kriging_sal = kriging_sal_flat.reshape(lon_mesh.shape)
    return kriging_temp, kriging_sal, lon_grid, lat_grid, depth_grid
def validate_kriging(kriging_temp, kriging_sal, lon_grid, lat_grid, depth_grid, argo_df):
    temp_interp = RegularGridInterpolator(
        (lon_grid, lat_grid, depth_grid), kriging_temp,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    sal_interp = RegularGridInterpolator(
        (lon_grid, lat_grid, depth_grid), kriging_sal,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    eval_coords = argo_df[["LONGITUDE", "LATITUDE", "DEPTH"]].values
    pred_t = temp_interp(eval_coords)
    pred_s = sal_interp(eval_coords)
    argo_valid = argo_df.copy()
    argo_valid["PRED_TEMP"] = pred_t
    argo_valid["PRED_SAL"] = pred_s
    argo_valid = argo_valid.dropna(subset=["PRED_TEMP", "PRED_SAL"])
    metrics_temp = calc_metrics(argo_valid["TEMP"].values, argo_valid["PRED_TEMP"].values)
    metrics_sal = calc_metrics(argo_valid["PSAL"].values, argo_valid["PRED_SAL"].values)
    print("\n=== Kriging 验证指标 ===")
    print(f"温度 - RMSE: {metrics_temp['RMSE']:.3f}℃, R²: {metrics_temp['R²']:.3f}")
    print(f"盐度 - RMSE: {metrics_sal['RMSE']:.3f}psu, R²: {metrics_sal['R²']:.3f}")
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
    line1, = ax1.plot(prof_t_pred, depth_grid, color=color_t, linewidth=4.0, label='Kriging Pred Temp')
    scatter1 = ax1.scatter(argo_profile["TEMP"], argo_profile["DEPTH"],
                           color=color_t, marker='o', facecolors='white', edgecolors=color_t, s=100, linewidth=2.0,
                           label='Argo Obs Temp', zorder=5)
    ax1.tick_params(axis='x', labelcolor=color_t, labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.invert_yaxis()  # 深度向下
    ax1.grid(alpha=0.4, linewidth=1.5)
    ax2 = ax1.twiny()
    color_s = 'tab:red'
    ax2.set_xlabel('Salinity (psu)', color=color_s, fontsize=20, fontweight='bold')
    line2, = ax2.plot(prof_s_pred, depth_grid, color=color_s, linewidth=4.0, linestyle='--',
                      label='Kriging Pred Salt')
    scatter2 = ax2.scatter(argo_profile["PSAL"], argo_profile["DEPTH"],
                           color=color_s, marker='x', s=100, linewidth=2.5, label='Argo Obs Salt', zorder=5)
    ax2.tick_params(axis='x', labelcolor=color_s, labelsize=16)
    lines = [line1, scatter1, line2, scatter2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=16, frameon=True, shadow=True, framealpha=0.9)
    plt.title(f"Vertical Profile Comparison (Kriging)\nLocation: {nearest_lon:.2f}°E, {nearest_lat:.2f}°N",
              fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig(f"{config['fig_save_dir']}Vertical_Profile.png", dpi=300)
    plt.close()

def plot_layered_error_optimized(argo_valid):
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
    plt.title("Layered RMSE Statistics (Kriging) (Precision: .3f)", fontsize=22, pad=15)
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
    print("=== 开始单独 Kriging 插值实验 ===")
    argo_df = load_argo_data_for_kriging()
    kriging_temp, kriging_sal, lon_grid, lat_grid, depth_grid = run_kriging_3d(argo_df)
    argo_valid, metrics_temp, metrics_sal = validate_kriging(
        kriging_temp, kriging_sal, lon_grid, lat_grid, depth_grid, argo_df
    )
    save_metrics_to_csv(metrics_temp, metrics_sal, config["fig_save_dir"])
    print("\n正在生成可视化图表...")
    plot_numerical_consistency(argo_valid, metrics_temp, metrics_sal)
    plot_spatial_distribution(kriging_temp, kriging_sal, lon_grid, lat_grid, depth_grid, argo_valid)
    plot_vertical_profile(kriging_temp, kriging_sal, lon_grid, lat_grid, depth_grid, argo_valid)
    plot_layered_error_optimized(argo_valid)
    print(f"\n=== 全部完成，结果已保存至 {config['fig_save_dir']} ===")