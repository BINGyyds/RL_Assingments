import pandas as pd
import matplotlib.pyplot as plt

# --- 参数配置 ---
# 你的两个CSV文件名
FILE_ORIGINAL = 'rewards_original_dqn.csv'
FILE_TARGET = 'rewards_target_dqn.csv'

# 滑动平均的窗口大小。这个值越大，曲线越平滑。50或100是比较常用的值。
WINDOW_SIZE = 50

# 图片保存路径
OUTPUT_FILE = 'dqn_comparison_plot.png'

# --- 代码主体 ---

def plot_comparison():
    """
    读取两个CSV文件，计算滑动平均奖励，并绘制对比图。
    """
    try:
        # 读取数据
        df_original = pd.read_csv(FILE_ORIGINAL)
        df_target = pd.read_csv(FILE_TARGET)
        print(f"成功读取 '{FILE_ORIGINAL}' (共 {len(df_original)} 个回合)")
        print(f"成功读取 '{FILE_TARGET}' (共 {len(df_target)} 个回合)")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保文件名正确，且文件与脚本在同一目录下。")
        return

    # 计算滑动平均奖励
    # .rolling() 创建一个滑动窗口, .mean() 计算窗口内的平均值
    df_original['smoothed_reward'] = df_original['total_reward'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    df_target['smoothed_reward'] = df_target['total_reward'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    
    # 开始绘图
    plt.style.use('seaborn-v0_8-whitegrid') # 使用一个美观的绘图风格
    plt.figure(figsize=(12, 7)) # 设置图片大小

    # 绘制 "DQN with Target Network" 的曲线
    # 先用半透明的浅色绘制原始数据点
    plt.plot(df_target['episode'], df_target['total_reward'], color='dodgerblue', alpha=0.2)
    # 再用深色绘制平滑后的曲线
    plt.plot(df_target['episode'], df_target['smoothed_reward'], color='dodgerblue', 
             label=f'DQN with Target Network (Avg over {WINDOW_SIZE} episodes)')

    # 绘制 "Original DQN" 的曲线
    # 先用半透明的浅色绘制原始数据点
    plt.plot(df_original['episode'], df_original['total_reward'], color='darkorange', alpha=0.2)
    # 再用深色绘制平滑后的曲线
    plt.plot(df_original['episode'], df_original['smoothed_reward'], color='darkorange', 
             label=f'Original DQN (Avg over {WINDOW_SIZE} episodes)')

    # 添加图表元素
    plt.title('DQN Performance Comparison on Flappy Bird', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward per Episode', fontsize=12)
    plt.legend(fontsize=12) # 显示图例
    
    # 保存图片
    plt.savefig(OUTPUT_FILE, dpi=300) # dpi=300 保证图片高清
    print(f"对比图已成功保存到 '{OUTPUT_FILE}'")

    # 显示图片
    plt.show()


if __name__ == '__main__':
    plot_comparison()