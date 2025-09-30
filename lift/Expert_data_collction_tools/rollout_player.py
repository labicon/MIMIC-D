import pickle as pkl
import matplotlib.pyplot as plt
import os
import time
import glob
import re

# -----------------
# 全局控制标志
# -----------------
# 这些标志用于 Matplotlib 事件处理器和主循环之间的通信
global_next_file = False
global_mark_damaged = False
global_exit_program = False 

def on_key_press(event):
    """
    处理 Matplotlib 窗口中的按键事件，设置全局控制标志。
    """
    global global_next_file, global_mark_damaged, global_exit_program
    
    # 强制 Matplotlib 立即处理事件
    try:
        plt.gcf().canvas.flush_events() 
    except Exception:
        pass # 如果窗口已经关闭，忽略错误

    key = event.key.lower()

    if key in ('n', 'right'):
        global_next_file = True
        print("➡️ Switching to next file...")
    
    elif key == 'd':
        global_mark_damaged = True
        print("⚠️ Current file marked as damaged.")

    elif key in ('escape', 'q'):
        global_exit_program = True
        plt.close('all')
        print("🛑 Exiting player.")
        
# -----------------
# RolloutPlayer 类
# -----------------

class RolloutPlayer:
    """
    加载并连续播放单个 .pkl 文件中的图像。
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.rollout = None
        self.fig = None
        self.im0 = None
        self.im1 = None
        self.current_frame = 0
        self.num_frames = 0
        self.load_data()

    def load_data(self):
        """加载 .pkl 文件中的数据。"""
        try:
            if not os.path.exists(self.filepath):
                 raise FileNotFoundError(f"File not found at {self.filepath}")

            with open(self.filepath, 'rb') as f:
                self.rollout = pkl.load(f)
            
            if 'camera0_obs' not in self.rollout or not self.rollout['camera0_obs']:
                raise ValueError("'camera0_obs' missing or empty.")
            if 'camera1_obs' not in self.rollout or not self.rollout['camera1_obs']:
                raise ValueError("'camera1_obs' missing or empty.")
            
            self.num_frames = min(len(self.rollout['camera0_obs']), len(self.rollout['camera1_obs']))
            self.current_frame = 0

        except Exception as e:
            print(f"An error occurred while loading {os.path.basename(self.filepath)}: {e}")
            self.rollout = None

    def setup_display(self):
        """设置 Matplotlib 图形界面并连接按键事件。"""
        if self.rollout is None:
            return

        plt.ion()
        self.fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示第一帧图像
        self.im0 = ax0.imshow(self.rollout['camera0_obs'][0])
        ax0.set_title(f"Robot 0 Camera: {os.path.basename(self.filepath)} | Frame 0/{self.num_frames}")
        ax0.axis('off')
        
        self.im1 = ax1.imshow(self.rollout['camera1_obs'][0])
        ax1.set_title("Robot 1 Camera Image")
        ax1.axis('off')
        
        self.fig.tight_layout()
        
        # 关键：连接按键事件到当前图形实例
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        plt.show(block=False)

    def update_frame(self):
        """更新当前显示的图像帧。"""
        if self.rollout is None or self.fig is None:
            return False

        if 0 <= self.current_frame < self.num_frames:
            self.im0.set_data(self.rollout['camera0_obs'][self.current_frame])
            self.im1.set_data(self.rollout['camera1_obs'][self.current_frame])
            
            # 更新标题中的帧数
            self.fig.axes[0].set_title(f"Robot 0 Camera: {os.path.basename(self.filepath)} | Frame {self.current_frame+1}/{self.num_frames}")
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.current_frame += 1
            return True 
        else:
            return False # 播放完毕

    def play_loop(self):
        """连续播放图像帧，直到播放完毕或被按键中断。"""
        global global_next_file, global_mark_damaged, global_exit_program

        if self.rollout is None:
            return

        self.setup_display()

        is_playing = True
        # 循环条件：仍在播放，窗口打开，且没有中断标志
        while is_playing and plt.get_fignums() and not (global_next_file or global_mark_damaged or global_exit_program):
            
            is_playing = self.update_frame()
            
            # 短暂暂停并处理 Matplotlib 事件
            plt.pause(0.01) 
            
            # 再次检查标志，如果按键发生在 plt.pause 期间，立即跳出
            if global_next_file or global_mark_damaged or global_exit_program:
                break

        # 退出时清理图形窗口
        if self.fig is not None:
             plt.close(self.fig)
        
# -------------------------
# 文件处理和主程序流程
# -------------------------

def natural_sort_key(s):
    """用于自然排序的键函数，提取文件名中的数字进行排序。"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def process_rollouts_in_directory(directory_path, file_pattern="rollout_seed*_mode2.pkl"):
    """
    遍历文件夹中的 .pkl 文件，并允许用户通过按键控制播放和标记。
    """
    global global_next_file, global_mark_damaged, global_exit_program

    # 1. 文件获取和自然排序
    search_path = os.path.join(directory_path, file_pattern)
    all_files = glob.glob(search_path)
    
    # 使用自然排序和过滤已损坏的文件
    all_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
    files_to_process = [f for f in all_files if "_damaged.pkl" not in os.path.basename(f)]

    if not files_to_process:
        print(f"No unprocessed files matching '{file_pattern}' found in directory: {directory_path}")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    plt.close('all') 

    # ------------------------------------------------
    # 修复：使用 while 循环和索引控制，解决跳过问题
    # ------------------------------------------------
    i = 0
    while i < len(files_to_process):
        file_path = files_to_process[i]
        
        if global_exit_program:
            break
        
        # --- 1. 处理标记损坏的逻辑 ('d' 键) ---
        if global_mark_damaged:
            global_mark_damaged = False
            
            # 安全重命名逻辑
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            base, ext = os.path.splitext(filename)
            new_filename = base + "_damaged" + ext
            new_file_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(file_path, new_file_path)
                print(f"✅ Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"❌ Error renaming file {filename}: {e}")
            
            # 文件已处理，移动到下一个文件
            i += 1 
            continue 
        
        # --- 2. 处理跳过下一文件 ('n' 键) ---
        if global_next_file:
            # global_next_file=True 意味着上一个文件被中断了。
            # 这里重置标志，并递增索引，跳过当前文件。
            global_next_file = False
            i += 1
            continue


        # --- 3. 播放当前文件 ---
        print(f"\n▶️ Now playing: {os.path.basename(file_path)}")

        player = RolloutPlayer(file_path)
        
        if player.rollout is None:
            i += 1 # 加载失败也算处理完毕，看下一个
            continue 

        # 开始播放。如果被 'n', 'd', 'q' 中断，循环会立即退出。
        player.play_loop() 

        # --- 4. 播放结束后，处理索引 ---
        
        if global_exit_program:
            break
        
        # 如果既没有按 'd' 也没有按 'n' 中断 (即播放完毕)，则移动到下一个文件。
        # 如果是按 'd' 或 'n' 中断，i 保持不变，下一次循环会在上面两个 if 语句中处理。
        if not global_mark_damaged and not global_next_file:
            i += 1
            
    print("\n🏁 All files processed.")


# --- 示例用法 ---
# ⚠️ 注意：file_pattern 默认为 "rollout_seed*_mode2.pkl"，
# 请根据您的实际文件命名情况进行修改。
directory = "../rollouts/newslower" 
process_rollouts_in_directory(directory, file_pattern="rollout_seed*_mode3.pkl")