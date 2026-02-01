import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.io import wavfile
import sounddevice as sd
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import messagebox
import copy
import time
import threading

# -------------------------
# CONFIG
# -------------------------
INPUT_AUDIO = "input.mp3"
FIXED_AUDIO = "input_fixed.wav"
OUTPUT_DIR = "output_slices"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# FIND FFMPEG
# -------------------------
def find_ffmpeg():
    """Find ffmpeg executable"""
    possible_paths = [
        r"D:\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
        "ffmpeg",
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
    ]
    
    for path in possible_paths:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            return path
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    return None

# -------------------------
# FIX CORRUPTED MP3
# -------------------------
print("Step 1: Finding FFmpeg...")
print("-" * 50)

ffmpeg_path = find_ffmpeg()

if ffmpeg_path is None:
    print("❌ FFmpeg not found!")
    exit(1)

print(f"✓ Found FFmpeg at: {ffmpeg_path}")

print("\nStep 2: Fixing/Converting audio with FFmpeg...")
print("-" * 50)

try:
    result = subprocess.run([
        ffmpeg_path,
        "-i", INPUT_AUDIO,
        "-ar", "44100",
        "-ac", "1",
        "-y",
        FIXED_AUDIO
    ], capture_output=True, text=True, check=True)
    
    print(f"✓ Fixed audio saved as: {FIXED_AUDIO}")
    
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg error: {e.stderr}")
    exit(1)

# -------------------------
# LOAD AUDIO
# -------------------------
print("\nStep 3: Loading audio...")
print("-" * 50)

sr, audio = wavfile.read(FIXED_AUDIO)

# Convert to float and normalize
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0
elif audio.dtype == np.uint8:
    audio = (audio.astype(np.float32) - 128) / 128.0
else:
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

audio = np.clip(audio, -1.0, 1.0)

duration = len(audio) / sr
print(f"✓ Loaded {duration:.2f} seconds of audio at {sr} Hz")

# -------------------------
# INTERACTIVE SLICER
# -------------------------
class AudioSlicer:
    def __init__(self, audio, sr, output_dir):
        self.audio = audio
        self.sr = sr
        self.duration = len(audio) / sr
        self.time = np.linspace(0, self.duration, len(audio))
        self.output_dir = output_dir
        self.cut_positions = []
        self.exclusion_zones = []
        self.playing = False
        self.paused = False
        self.play_start_time = 0
        self.pause_time = 0
        self.playback_line = None
        self.playback_thread = None
        
        # Undo stack
        self.history = []
        self.max_history = 50
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(18, 7))
        plt.subplots_adjust(bottom=0.15)
        
        # Plot waveform
        self.waveform_line, = self.ax.plot(self.time, self.audio, linewidth=0.4, color='steelblue')
        self.ax.set_title('Interactive Audio Slicer - Press "Help" for instructions', 
                         fontsize=13, fontweight='bold', pad=20)
        self.ax.set_xlabel('Time (seconds)', fontsize=11)
        self.ax.set_ylabel('Amplitude', fontsize=11)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.grid(True, alpha=0.3)
        
        # Storage for visual elements
        self.cut_lines = []
        self.exclusion_patches = []
        
        # Add buttons
        ax_help = plt.axes([0.12, 0.02, 0.06, 0.05])
        ax_undo = plt.axes([0.19, 0.02, 0.06, 0.05])
        ax_clear = plt.axes([0.26, 0.02, 0.06, 0.05])
        ax_save = plt.axes([0.33, 0.02, 0.06, 0.05])
        ax_play = plt.axes([0.50, 0.02, 0.06, 0.05])
        ax_pause = plt.axes([0.57, 0.02, 0.06, 0.05])
        ax_stop = plt.axes([0.64, 0.02, 0.06, 0.05])
        
        self.btn_help = Button(ax_help, 'Help', color='lightblue', hovercolor='dodgerblue')
        self.btn_undo = Button(ax_undo, 'Undo', color='lightyellow', hovercolor='gold')
        self.btn_clear = Button(ax_clear, 'Clear', color='lightcoral', hovercolor='red')
        self.btn_save = Button(ax_save, 'Save', color='lightgreen', hovercolor='green')
        self.btn_play = Button(ax_play, 'Play', color='palegreen', hovercolor='limegreen')
        self.btn_pause = Button(ax_pause, 'Pause', color='wheat', hovercolor='orange')
        self.btn_stop = Button(ax_stop, 'Stop', color='lightgray', hovercolor='darkgray')
        
        self.btn_help.on_clicked(self.show_help)
        self.btn_undo.on_clicked(self.undo_action)
        self.btn_clear.on_clicked(self.clear_cuts)
        self.btn_save.on_clicked(self.save_slices)
        self.btn_play.on_clicked(self.play_audio)
        self.btn_pause.on_clicked(self.pause_audio)
        self.btn_stop.on_clicked(self.stop_audio)
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Info text
        self.info_text = self.ax.text(0.02, 0.98, 'Cuts: 0 | Slices: 1 | Excluded: 0', 
                                      transform=self.ax.transAxes, 
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                      fontsize=10)
        
        # Initialize Tkinter root for dialogs
        self.root = tk.Tk()
        self.root.withdraw()
        
        print("\n" + "="*60)
        print("INTERACTIVE AUDIO SLICER READY")
        print("="*60)
        print("Click the 'Help' button for instructions")
        print("="*60 + "\n")
    
    def save_state(self):
        """Save current state to history for undo"""
        state = {
            'cut_positions': copy.deepcopy(self.cut_positions),
            'exclusion_zones': copy.deepcopy(self.exclusion_zones)
        }
        self.history.append(state)
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def undo_action(self, event):
        """Undo last action"""
        if len(self.history) == 0:
            print("⚠ Nothing to undo")
            return
        
        state = self.history.pop()
        
        for line in self.cut_lines:
            line.remove()
        for patch in self.exclusion_patches:
            patch.remove()
        
        self.cut_lines = []
        self.exclusion_patches = []
        
        self.cut_positions = state['cut_positions']
        self.exclusion_zones = state['exclusion_zones']
        
        for cut_time in self.cut_positions:
            line = self.ax.axvline(cut_time, color='red', linestyle='--', 
                                  linewidth=2, alpha=0.7)
            self.cut_lines.append(line)
        
        for start, end in self.exclusion_zones:
            rect = Rectangle((start, -1.1), end - start, 2.2, 
                           facecolor='red', alpha=0.2, edgecolor='darkred', linewidth=1)
            self.ax.add_patch(rect)
            self.exclusion_patches.append(rect)
        
        self.update_display()
        print("↶ Undone last action")
    
    def show_help(self, event):
        """Show help dialog"""
        help_text = """INTERACTIVE AUDIO SLICER - INSTRUCTIONS

BASIC CONTROLS:
• LEFT-CLICK on waveform → Add a cut point
• RIGHT-CLICK on cut line → Remove that cut
• RIGHT-CLICK on segment → Exclude that segment
• SPACE (alone) → Play/Pause audio
• SPACE + CLICK → Seek to position and play
• CTRL + Z → Undo last action

NAVIGATION:
• CTRL + SCROLL → Zoom in/out
• SHIFT + SCROLL → Pan left/right

BUTTONS:
• Help → Show this help dialog
• Undo → Undo last action (also Ctrl+Z)
• Clear → Remove all cuts and exclusions
• Save → Export audio segments
• Play/Pause/Stop → Audio playback controls

VISUAL INDICATORS:
• Red dashed lines → Cut points
• Red shaded areas → Excluded segments (won't be saved)
• Green line → Current playback position

WORKFLOW:
1. Add cut points by left-clicking on the waveform
2. Right-click on cut lines to remove them
3. Right-click on segments to exclude them
4. Use Space to preview audio playback
5. Click 'Save' to export the slices

TIPS:
• Right-click close to a cut line to remove it
• Right-click in the middle of a segment to exclude it
• Excluded segments are skipped during export
• Use Ctrl+Z to undo mistakes
• Zoom in for precise cut placement"""
        
        messagebox.showinfo("Help - Audio Slicer", help_text)
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'ctrl+z':
            self.undo_action(None)
        elif event.key == ' ' and event.inaxes == self.ax:
            if self.playing and not self.paused:
                self.pause_audio(None)
            elif self.paused:
                self.play_audio(None)
            elif not self.playing:
                self.play_audio(None)
    
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        xlim = self.ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        
        if event.key == 'control':
            zoom_factor = 1.2 if event.button == 'down' else 0.8
            new_range = xrange * zoom_factor
            
            mouse_x = event.xdata
            left_portion = (mouse_x - xlim[0]) / xrange
            right_portion = (xlim[1] - mouse_x) / xrange
            
            new_xlim = [
                mouse_x - new_range * left_portion,
                mouse_x + new_range * right_portion
            ]
            
            new_xlim[0] = max(0, new_xlim[0])
            new_xlim[1] = min(self.duration, new_xlim[1])
            
            if new_xlim[0] >= new_xlim[1]:
                return
            
            self.ax.set_xlim(new_xlim)
            self.fig.canvas.draw_idle()
        
        elif event.key == 'shift':
            pan_amount = xrange * 0.1
            
            if event.button == 'up':
                pan_amount = -pan_amount
            
            new_xlim = [xlim[0] + pan_amount, xlim[1] + pan_amount]
            
            if new_xlim[0] < 0:
                new_xlim = [0, xrange]
            if new_xlim[1] > self.duration:
                new_xlim = [self.duration - xrange, self.duration]
            
            self.ax.set_xlim(new_xlim)
            self.fig.canvas.draw_idle()
    
    def find_segment_at_position(self, position):
        """Find which segment contains the position"""
        boundaries = [0] + sorted(self.cut_positions) + [self.duration]
        
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= position <= boundaries[i + 1]:
                return (boundaries[i], boundaries[i + 1])
        
        return None
    
    def find_nearest_cut(self, position, threshold=None):
        """Find nearest cut to position"""
        if threshold is None:
            xlim = self.ax.get_xlim()
            visible_range = xlim[1] - xlim[0]
            threshold = visible_range * 0.015  # 1.5% of visible range
        
        if len(self.cut_positions) == 0:
            return None, float('inf')
        
        distances = [abs(cut_pos - position) for cut_pos in self.cut_positions]
        min_idx = distances.index(min(distances))
        min_dist = distances[min_idx]
        
        if min_dist < threshold:
            return min_idx, min_dist
        
        return None, float('inf')
    
    def is_segment_excluded(self, start, end):
        """Check if a segment is already excluded"""
        for ex_start, ex_end in self.exclusion_zones:
            if abs(ex_start - start) < 0.01 and abs(ex_end - end) < 0.01:
                return True
        return False
    
    def show_context_menu(self, position):
        """Automated right-click action: remove cut or exclude segment"""
        # First, check if we're near a cut line
        cut_idx, cut_dist = self.find_nearest_cut(position)
        
        if cut_idx is not None:
            # We're near a cut - remove it
            self.save_state()
            removed_pos = self.cut_positions.pop(cut_idx)
            self.cut_lines[cut_idx].remove()
            self.cut_lines.pop(cut_idx)
            self.update_display()
            print(f"✓ Removed cut at {removed_pos:.2f}s")
            return
        
        # Not near a cut, so exclude the segment we're in
        segment = self.find_segment_at_position(position)
        
        if not segment:
            print("⚠ Could not determine segment at this position")
            return
        
        # Check if this segment is already excluded
        is_excluded = self.is_segment_excluded(segment[0], segment[1])
        
        if is_excluded:
            # Un-exclude it
            self.save_state()
            for i, (ex_start, ex_end) in enumerate(self.exclusion_zones):
                if abs(ex_start - segment[0]) < 0.01 and abs(ex_end - segment[1]) < 0.01:
                    self.exclusion_zones.pop(i)
                    self.exclusion_patches[i].remove()
                    self.exclusion_patches.pop(i)
                    break
            self.update_display()
            print(f"✓ Included segment {segment[0]:.2f}s - {segment[1]:.2f}s")
        else:
            # Exclude it
            self.save_state()
            self.exclusion_zones.append((segment[0], segment[1]))
            
            rect = Rectangle((segment[0], -1.1), segment[1] - segment[0], 2.2, 
                           facecolor='red', alpha=0.2, edgecolor='darkred', linewidth=1)
            self.ax.add_patch(rect)
            self.exclusion_patches.append(rect)
            
            self.update_display()
            print(f"✓ Excluded segment {segment[0]:.2f}s - {segment[1]:.2f}s")
    
    def execute_action(self, action, cut_idx, segment):
        """This method is no longer needed - keeping for compatibility"""
        pass
    
    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        
        if event.key == ' ':
            seek_time = max(0, min(event.xdata, self.duration))
            
            sd.stop()
            self.playing = False
            self.paused = False
            
            if self.playback_line:
                self.playback_line.remove()
                self.playback_line = None
            
            print(f"⏩ Seeking to {seek_time:.2f}s and playing...")
            self.play_start_time = seek_time
            
            self.playback_line = self.ax.axvline(seek_time, color='green', linestyle='-', 
                                                  linewidth=2, alpha=0.8, label='Playing')
            
            remaining_audio = self.audio[int(seek_time * self.sr):]
            self.playing = True
            sd.play(remaining_audio, self.sr)
            
            self.playback_thread = threading.Thread(target=self.update_playback_position)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            self.fig.canvas.draw_idle()
            return
        
        if event.button == 3:
            # Right-click: automated action
            self.show_context_menu(event.xdata)
            return
        
        if event.button == 1:
            # Left-click: add cut
            self.save_state()
            
            cut_time = event.xdata
            self.cut_positions.append(cut_time)
            self.cut_positions.sort()
            
            line = self.ax.axvline(cut_time, color='red', linestyle='--', 
                                  linewidth=2, alpha=0.7)
            
            insert_idx = self.cut_positions.index(cut_time)
            self.cut_lines.insert(insert_idx, line)
            
            self.update_display()
            print(f"✓ Added cut at {cut_time:.2f}s")
    
    def update_display(self):
        num_cuts = len(self.cut_positions)
        num_slices = num_cuts + 1 - len(self.exclusion_zones)
        num_excluded = len(self.exclusion_zones)
        self.info_text.set_text(f'Cuts: {num_cuts} | Active Slices: {num_slices} | Excluded: {num_excluded}')
        self.fig.canvas.draw_idle()
    
    def clear_cuts(self, event):
        if len(self.cut_positions) > 0 or len(self.exclusion_zones) > 0:
            self.save_state()
        
        for line in self.cut_lines:
            line.remove()
        for patch in self.exclusion_patches:
            patch.remove()
        
        self.cut_positions = []
        self.cut_lines = []
        self.exclusion_zones = []
        self.exclusion_patches = []
        self.update_display()
        print("✓ Cleared all cuts and exclusions")
    
    def play_audio(self, event):
        if self.playing and not self.paused:
            print("⚠ Audio already playing")
            return
        
        try:
            if self.paused:
                print("▶ Resuming audio...")
                remaining_audio = self.audio[int(self.pause_time * self.sr):]
                self.play_start_time = self.pause_time
                self.paused = False
                self.playing = True
                
                sd.play(remaining_audio, self.sr)
                
                self.playback_thread = threading.Thread(target=self.update_playback_position)
                self.playback_thread.daemon = True
                self.playback_thread.start()
            else:
                print("▶ Playing audio...")
                self.playing = True
                self.paused = False
                self.play_start_time = 0
                
                if self.playback_line:
                    self.playback_line.remove()
                self.playback_line = self.ax.axvline(0, color='green', linestyle='-', 
                                                      linewidth=2, alpha=0.8, label='Playing')
                
                sd.play(self.audio, self.sr)
                
                self.playback_thread = threading.Thread(target=self.update_playback_position)
                self.playback_thread.daemon = True
                self.playback_thread.start()
                
            print("  (Press Space or click 'Pause' to pause)")
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            self.playing = False
            self.paused = False
    
    def update_playback_position(self):
        start_time = time.time()
        
        while self.playing and not self.paused:
            elapsed = time.time() - start_time
            current_pos = self.play_start_time + elapsed
            
            if current_pos >= self.duration:
                self.playing = False
                if self.playback_line:
                    self.playback_line.remove()
                    self.playback_line = None
                    self.fig.canvas.draw_idle()
                break
            
            if self.playback_line:
                self.playback_line.set_xdata([current_pos, current_pos])
                self.fig.canvas.draw_idle()
            
            time.sleep(0.05)
        
        if not self.playing or self.paused:
            if self.playback_line and not self.paused:
                self.playback_line.remove()
                self.playback_line = None
                self.fig.canvas.draw_idle()
    
    def pause_audio(self, event):
        if not self.playing:
            print("⚠ No audio is playing")
            return
        
        if self.paused:
            print("⚠ Audio already paused")
            return
        
        if self.playback_line:
            self.pause_time = self.playback_line.get_xdata()[0]
        else:
            self.pause_time = 0
        
        sd.stop()
        self.paused = True
        self.playing = False
        print(f"⏸ Paused at {self.pause_time:.2f}s")
    
    def stop_audio(self, event):
        if self.playing or self.paused:
            sd.stop()
            self.playing = False
            self.paused = False
            self.play_start_time = 0
            self.pause_time = 0
            
            if self.playback_line:
                self.playback_line.remove()
                self.playback_line = None
                self.fig.canvas.draw_idle()
            
            print("■ Stopped playback")
        else:
            print("⚠ No audio is playing")
    
    def save_slices(self, event):
        if len(self.cut_positions) == 0:
            print("\n⚠ No cuts defined. Checking for exclusions...")
            
            if len(self.exclusion_zones) == 1 and \
               abs(self.exclusion_zones[0][0]) < 0.01 and \
               abs(self.exclusion_zones[0][1] - self.duration) < 0.01:
                print("❌ Entire audio is excluded. Nothing to save.")
                return
            elif len(self.exclusion_zones) == 0:
                chunks = [self.audio]
                chunk_times = [(0, self.duration)]
            else:
                print("❌ Cannot process exclusions without cut points.")
                print("    Add cut points to define segments first.")
                return
        else:
            chunks = []
            chunk_times = []
            
            boundaries = [0] + sorted(self.cut_positions) + [len(self.audio) / self.sr]
            
            for i in range(len(boundaries) - 1):
                start_time = boundaries[i]
                end_time = boundaries[i + 1]
                
                is_excluded = False
                for ex_start, ex_end in self.exclusion_zones:
                    if abs(start_time - ex_start) < 0.01 and abs(end_time - ex_end) < 0.01:
                        is_excluded = True
                        break
                
                if not is_excluded:
                    start_sample = int(start_time * self.sr)
                    end_sample = int(end_time * self.sr)
                    chunk = self.audio[start_sample:end_sample]
                    chunks.append(chunk)
                    chunk_times.append((start_time, end_time))
        
        if len(chunks) == 0:
            print("\n❌ No segments to save. All segments are excluded.")
            return
        
        print("\n" + "="*60)
        print("EXPORTING SLICES")
        print("="*60)
        
        for i, (chunk, (start_time, end_time)) in enumerate(zip(chunks, chunk_times)):
            filename = f"slice_{i+1:03d}.wav"
            output_path = os.path.join(self.output_dir, filename)
            
            chunk_int16 = (chunk * 32767).astype(np.int16)
            wavfile.write(output_path, self.sr, chunk_int16)
            
            duration_sec = len(chunk) / self.sr
            print(f"  ✓ {filename}: {start_time:.2f}s - {end_time:.2f}s ({duration_sec:.2f}s)")
        
        print("="*60)
        print(f"✓ Saved {len(chunks)} slices to '{self.output_dir}/'")
        if len(self.exclusion_zones) > 0:
            print(f"  ({len(self.exclusion_zones)} segments excluded)")
        print("="*60)
    
    def show(self):
        plt.show()

# Create and show slicer
slicer = AudioSlicer(audio, sr, OUTPUT_DIR)
slicer.show()