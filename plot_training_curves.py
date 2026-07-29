"""
Script trực quan hóa đồ thị biến thiên Loss và Dice Score theo từng Epoch (Per-Epoch Training Curves)
====================================================================================================
Hỗ trợ tự động phát hiện và tách biệt nhiều lượt huấn luyện (Training Runs/Sessions) trong cùng 1 file CSV 
(ví dụ: 1 đợt train tập Only Tumor / Split và 1 đợt train tập Full Data).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình phong cách đồ thị đẹp mắt
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.2,
    "grid.color": "#eeeeee",
    "grid.linestyle": "--",
    "figure.dpi": 300
})


def find_log_files(checkpoint_dir="checkpoint"):
    """Dò tìm tất cả các file CSV log huấn luyện trong thư mục checkpoint"""
    if not os.path.exists(checkpoint_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(project_root, "checkpoint")

    if not os.path.exists(checkpoint_dir):
        return []

    csv_files = glob.glob(os.path.join(checkpoint_dir, "*.csv"))
    return csv_files


def split_csv_sessions(csv_path):
    """
    Tách 1 file CSV ra thành danh sách các lượt train (Sessions/Runs) riêng biệt 
    dựa trên tiêu đề '# Training Run Started At:' hoặc khi cột epoch bị reset về 0.
    """
    sessions = []
    current_lines = []
    current_time = "Unknown_Time"

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("# Training Run Started At:"):
            # Nếu đang có session dở dang, lưu lại trước khi sang session mới
            if current_lines:
                sessions.append((current_time, current_lines))
                current_lines = []
            current_time = line.replace("# Training Run Started At:", "").strip()
        elif line.startswith("#"):
            continue
        else:
            if line.strip():
                current_lines.append(line)

    if current_lines:
        sessions.append((current_time, current_lines))

    parsed_dfs = []
    for time_str, raw_lines in sessions:
        from io import StringIO
        try:
            df = pd.read_csv(StringIO("".join(raw_lines)))
            # Lọc bớt header lặp lại
            if 'epoch' in df.columns:
                df = df[pd.to_numeric(df['epoch'], errors='coerce').notnull()].copy()
                df['epoch'] = df['epoch'].astype(int)
                for col in df.columns:
                    if col != 'epoch':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Tách nếu trong session nhỏ có reset epoch về 0
                reset_indices = df[df['epoch'] == 0].index.tolist()
                if len(reset_indices) > 1:
                    for i in range(len(reset_indices)):
                        start_idx = reset_indices[i]
                        end_idx = reset_indices[i+1] if i + 1 < len(reset_indices) else len(df)
                        sub_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
                        if not sub_df.empty:
                            parsed_dfs.append((time_str, sub_df))
                else:
                    df = df.sort_values(by='epoch').reset_index(drop=True)
                    if not df.empty:
                        parsed_dfs.append((time_str, df))
        except Exception as e:
            continue

    return parsed_dfs


def plot_single_session_history(model_name, dataset_tag, session_idx, time_str, df, output_dir):
    """Vẽ đồ thị chi tiết Loss và Dice per-Epoch cho 1 lượt train cụ thể"""
    if df.empty or 'loss' not in df.columns or 'val_loss' not in df.columns:
        return None

    epochs = df['epoch'] + 1  # 1-indexed epochs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 1. ĐỒ THỊ LOSS ---
    axes[0].plot(epochs, df['loss'], label='Train Loss', color='#1f77b4', linewidth=2.2)
    axes[0].plot(epochs, df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2.2, linestyle='--')
    
    min_val_loss_idx = df['val_loss'].idxmin()
    best_loss_epoch = epochs.iloc[min_val_loss_idx]
    best_val_loss = df['val_loss'].iloc[min_val_loss_idx]
    axes[0].scatter(best_loss_epoch, best_val_loss, color='red', s=70, zorder=5)
    axes[0].annotate(f'Best Val Loss: {best_val_loss:.4f} (Ep {best_loss_epoch})',
                    (best_loss_epoch, best_val_loss),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=9, fontweight='bold', color='#d62728')

    axes[0].set_title(f"Hanh trinh Loss — {model_name} ({dataset_tag.upper()})", fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)

    # --- 2. ĐỒ THỊ DICE COEFFICIENT ---
    best_val_dice = 0.0
    best_dice_epoch = 0

    if 'dice_coef' in df.columns and 'val_dice_coef' in df.columns:
        axes[1].plot(epochs, df['dice_coef'], label='Train Dice (Overall)', color='#2ca02c', linewidth=2.2)
        axes[1].plot(epochs, df['val_dice_coef'], label='Val Dice (Overall)', color='#d62728', linewidth=2.2, linestyle='--')

        max_val_dice_idx = df['val_dice_coef'].idxmax()
        best_dice_epoch = epochs.iloc[max_val_dice_idx]
        best_val_dice = df['val_dice_coef'].iloc[max_val_dice_idx]
        axes[1].scatter(best_dice_epoch, best_val_dice, color='green', s=70, zorder=5)
        axes[1].annotate(f'Best Val Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%) (Ep {best_dice_epoch})',
                        (best_dice_epoch, best_val_dice),
                        textcoords="offset points", xytext=(0, 10), ha='center',
                        fontsize=9, fontweight='bold', color='#2ca02c')

    if 'val_dice_benign' in df.columns:
        axes[1].plot(epochs, df['val_dice_benign'], label='Val Dice (Benign - U Lanh)', color='#9467bd', linewidth=1.5, linestyle=':')
    if 'val_dice_malignant' in df.columns:
        axes[1].plot(epochs, df['val_dice_malignant'], label='Val Dice (Malignant - U Ac)', color='#8c564b', linewidth=1.5, linestyle=':')

    axes[1].set_title(f"Hanh trinh Dice Score — {model_name} ({dataset_tag.upper()})", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Dice Coefficient", fontsize=11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc='lower right', frameon=True)

    plt.tight_layout()

    tag_suffix = f"_{dataset_tag}" if dataset_tag else ""
    session_suffix = f"_run{session_idx}" if session_idx > 1 else ""
    save_path = os.path.join(output_dir, f"curve_{model_name}{tag_suffix}{session_suffix}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"[OK] Da ve do thi Loss & Dice -> {save_path}")
    return {
        "model_name": model_name,
        "dataset": dataset_tag.upper(),
        "run_time": time_str,
        "total_epochs": len(df),
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "best_epoch": best_dice_epoch
    }


def plot_comparison_summary(all_runs, output_dir):
    """Vẽ đồ thị so sánh đối chiếu giữa các lượt train"""
    if len(all_runs) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = sns.color_palette("tab10", len(all_runs))

    summary_list = []

    for idx, run in enumerate(all_runs):
        model_label = f"{run['model_name']} [{run['dataset']}]"
        df = run['df']
        epochs = df['epoch'] + 1
        color = colors[idx]

        if 'val_loss' in df.columns:
            axes[0].plot(epochs, df['val_loss'], label=model_label, color=color, linewidth=2)
        if 'val_dice_coef' in df.columns:
            axes[1].plot(epochs, df['val_dice_coef'], label=model_label, color=color, linewidth=2)

        summary_list.append({
            "Model": run['model_name'],
            "Dataset Split": run['dataset'],
            "Started Time": run['run_time'],
            "Epochs": len(df),
            "Best Val Loss": f"{df['val_loss'].min():.4f}",
            "Best Val Dice": f"{df['val_dice_coef'].max():.4f} ({df['val_dice_coef'].max()*100:.2f}%)",
            "Best Epoch": int(epochs.iloc[df['val_dice_coef'].idxmax()]) if 'val_dice_coef' in df.columns else "N/A"
        })

    axes[0].set_title("SO SANH VAL LOSS GIUA CAC LUOT TRAIN", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation Loss", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)

    axes[1].set_title("SO SANH VAL DICE SCORE GIUA CAC LUOT TRAIN", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Dice Coefficient", fontsize=11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc='lower right', frameon=True)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "comparison_all_runs.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"\n[INFO] Da tao do thi SO SANH DOI CHIEU giua tat ca cac luot train -> {save_path}")

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        print("\n" + "="*80)
        print("BANG TONG KET KET QUA DAT DUOC CAO NHAT TREN TAP VALIDATION (PER RUN)")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoint")
    output_plots_dir = os.path.join(checkpoint_dir, "plots")

    os.makedirs(output_plots_dir, exist_ok=True)

    print(f"\n[INFO] Dang do tim cac file log CSV trong: {checkpoint_dir}")
    log_files = find_log_files(checkpoint_dir)

    if not log_files:
        print(f"[WARNING] Khong tim thay file log CSV nao trong thu muc '{checkpoint_dir}'.")
        print("Hay chay huan luyen 'python train.py' truoc de sinh file log CSV!")
        return

    print(f"[INFO] Tim thay {len(log_files)} file log CSV. Dang phan tich tung luot train (Sessions)...\n")

    all_runs = []

    for csv_file in log_files:
        filename = os.path.basename(csv_file)
        raw_name = filename.replace("training_logs_", "").replace(".csv", "")
        
        # Nhận diện dataset tag từ tên file (split vs full)
        if "full" in raw_name.lower():
            dataset_tag = "FULL"
            model_name = raw_name.lower().replace("_full", "").upper()
        elif "split" in raw_name.lower():
            dataset_tag = "ONLY_TUMOR_SPLIT"
            model_name = raw_name.lower().replace("_split", "").upper()
        else:
            dataset_tag = "DEFAULT"
            model_name = raw_name.upper()

        sessions = split_csv_sessions(csv_file)
        for idx, (time_str, df) in enumerate(sessions, 1):
            res = plot_single_session_history(model_name, dataset_tag, idx, time_str, df, output_plots_dir)
            if res:
                all_runs.append({
                    "model_name": model_name,
                    "dataset": dataset_tag,
                    "run_time": time_str,
                    "df": df
                })

    plot_comparison_summary(all_runs, output_plots_dir)


if __name__ == "__main__":
    main()
