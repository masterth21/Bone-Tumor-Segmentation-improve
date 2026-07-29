"""
Script trực quan hóa đồ thị biến thiên Loss và Dice Score theo từng Epoch (Per-Epoch Training Curves)
====================================================================================================
Tự động quét các file CSV log huấn luyện trong thư mục 'checkpoint/', vẽ và lưu biểu đồ sắc nét:
1. Đồ thị Loss (Train Loss vs Val Loss) qua từng Epoch.
2. Đồ thị Dice Coefficient tổng thể (Train Dice vs Val Dice) qua từng Epoch.
3. Đồ thị Dice từng lớp (U Lành tính vs U Ác tính) qua từng Epoch.
4. Biểu đồ so sánh xu hướng Loss & Dice giữa tất cả các mô hình đã huấn luyện.
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
        # Thử tìm relative từ project root
        project_root = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(project_root, "checkpoint")

    if not os.path.exists(checkpoint_dir):
        return []

    csv_files = glob.glob(os.path.join(checkpoint_dir, "*.csv"))
    return csv_files


def clean_dataframe(df):
    """Lọc bỏ các dòng tiêu đề trùng lặp do nới tiếp log nhiều lượt train"""
    # Nếu cột 'epoch' chứa chuỗi tiêu đề do ghi nối tiếp
    if 'epoch' in df.columns:
        df = df[pd.to_numeric(df['epoch'], errors='coerce').notnull()].copy()
        df['epoch'] = df['epoch'].astype(int)

    # Convert tất cả các cột số sang float
    for col in df.columns:
        if col != 'epoch':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(by='epoch').reset_index(drop=True)
    return df


def plot_single_model_history(csv_path, output_dir):
    """Vẽ đồ thị chi tiết Loss và Dice per-Epoch cho 1 mô hình"""
    filename = os.path.basename(csv_path)
    model_name = filename.replace("training_logs_", "").replace(".csv", "")

    try:
        # Đọc file CSV, bỏ qua các dòng ghi chú bắt đầu bằng '#'
        df = pd.read_csv(csv_path, comment='#')
        df = clean_dataframe(df)

        if df.empty or 'loss' not in df.columns or 'val_loss' not in df.columns:
            print(f"⚠️ File log {filename} không đủ dữ liệu để vẽ đồ thị.")
            return None

        epochs = df['epoch'] + 1  # 1-indexed epochs

        # Tạo Figure chứa 2 biểu đồ: Loss và Dice
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # --- 1. ĐỒ THỊ LOSS ---
        axes[0].plot(epochs, df['loss'], label='Train Loss', color='#1f77b4', linewidth=2.2)
        axes[0].plot(epochs, df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2.2, linestyle='--')
        
        # Đánh dấu epoch có Val Loss thấp nhất
        min_val_loss_idx = df['val_loss'].idxmin()
        best_loss_epoch = epochs.iloc[min_val_loss_idx]
        best_val_loss = df['val_loss'].iloc[min_val_loss_idx]
        axes[0].scatter(best_loss_epoch, best_val_loss, color='red', s=70, zorder=5)
        axes[0].annotate(f'Best Val Loss: {best_val_loss:.4f} (Ep {best_loss_epoch})',
                        (best_loss_epoch, best_val_loss),
                        textcoords="offset points", xytext=(0, 10), ha='center',
                        fontsize=9, fontweight='bold', color='#d62728')

        axes[0].set_title(f"Hành trình biến thiên Loss — {model_name}", fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].legend(loc='upper right', frameon=True)

        # --- 2. ĐỒ THỊ DICE COEFFICIENT ---
        if 'dice_coef' in df.columns and 'val_dice_coef' in df.columns:
            axes[1].plot(epochs, df['dice_coef'], label='Train Dice (Overall)', color='#2ca02c', linewidth=2.2)
            axes[1].plot(epochs, df['val_dice_coef'], label='Val Dice (Overall)', color='#d62728', linewidth=2.2, linestyle='--')

            # Đánh dấu epoch có Val Dice cao nhất
            max_val_dice_idx = df['val_dice_coef'].idxmax()
            best_dice_epoch = epochs.iloc[max_val_dice_idx]
            best_val_dice = df['val_dice_coef'].iloc[max_val_dice_idx]
            axes[1].scatter(best_dice_epoch, best_val_dice, color='green', s=70, zorder=5)
            axes[1].annotate(f'Best Val Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%) (Ep {best_dice_epoch})',
                            (best_dice_epoch, best_val_dice),
                            textcoords="offset points", xytext=(0, 10), ha='center',
                            fontsize=9, fontweight='bold', color='#2ca02c')

        # Thêm đường nét cho Dice Lành vs Ác nếu có
        if 'val_dice_benign' in df.columns:
            axes[1].plot(epochs, df['val_dice_benign'], label='Val Dice (Benign - U Lành)', color='#9467bd', linewidth=1.5, linestyle=':')
        if 'val_dice_malignant' in df.columns:
            axes[1].plot(epochs, df['val_dice_malignant'], label='Val Dice (Malignant - U Ác)', color='#8c564b', linewidth=1.5, linestyle=':')

        axes[1].set_title(f"Hành trình tăng trưởng Dice Score — {model_name}", fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Dice Coefficient", fontsize=11)
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend(loc='lower right', frameon=True)

        plt.tight_layout()

        # Lưu đồ thị
        save_path = os.path.join(output_dir, f"curve_{model_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"[OK] Da ve do thi Loss & Dice cho [{model_name}] -> {save_path}")
        return {
            "model_name": model_name,
            "total_epochs": len(df),
            "best_val_loss": best_val_loss,
            "best_val_loss_epoch": best_loss_epoch,
            "best_val_dice": best_val_dice,
            "best_val_dice_epoch": best_dice_epoch
        }

    except Exception as e:
        print(f"[ERROR] Loi khi doc/ve do thi file {filename}: {e}")
        return None


def plot_comparison_curves(log_files, output_dir):
    """Vẽ đồ thị đối chiếu so sánh Loss và Val Dice giữa TẤT CẢ các mô hình trên cùng 1 biểu đồ"""
    if len(log_files) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = sns.color_palette("tab10", len(log_files))

    summary_list = []

    for idx, csv_path in enumerate(log_files):
        filename = os.path.basename(csv_path)
        model_name = filename.replace("training_logs_", "").replace(".csv", "")

        try:
            df = pd.read_csv(csv_path, comment='#')
            df = clean_dataframe(df)
            if df.empty or 'val_loss' not in df.columns or 'val_dice_coef' not in df.columns:
                continue

            epochs = df['epoch'] + 1
            color = colors[idx]

            # 1. So sánh Val Loss qua từng Epoch
            axes[0].plot(epochs, df['val_loss'], label=f"{model_name}", color=color, linewidth=2)

            # 2. So sánh Val Dice qua từng Epoch
            axes[1].plot(epochs, df['val_dice_coef'], label=f"{model_name}", color=color, linewidth=2)

            summary_list.append({
                "Model": model_name,
                "Total Epochs": len(df),
                "Best Val Loss": f"{df['val_loss'].min():.4f}",
                "Best Val Dice": f"{df['val_dice_coef'].max():.4f} ({df['val_dice_coef'].max()*100:.2f}%)",
                "Best Epoch": int(epochs.iloc[df['val_dice_coef'].idxmax()])
            })

        except Exception as e:
            continue

    # Setup Trục Val Loss
    axes[0].set_title("SO SANH VAL LOSS GIUA CAC MO HINH", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation Loss", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)

    # Setup Trục Val Dice
    axes[1].set_title("SO SANH VAL DICE SCORE GIUA CAC MO HINH", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Dice Coefficient", fontsize=11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc='lower right', frameon=True)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "comparison_all_models.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"\n[INFO] Da tao do thi SO SANH DOI CHIEU giua cac mo hinh -> {save_path}")

    # In bảng tổng kết ra màn hình terminal
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        print("\n" + "="*70)
        print("BANG TONG KET KET QUA DAT DUOC CAO NHAT TREN TAP VALIDATION")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70 + "\n")


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

    print(f"[INFO] Tim thay {len(log_files)} file log CSV. Bat dau ve do thi per-Epoch...\n")

    summaries = []
    for csv_file in log_files:
        res = plot_single_model_history(csv_file, output_plots_dir)
        if res:
            summaries.append(res)

    # Vẽ đồ thị so sánh đối chiếu giữa các mô hình
    plot_comparison_curves(log_files, output_plots_dir)


if __name__ == "__main__":
    main()
