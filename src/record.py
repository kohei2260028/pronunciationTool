import os
import tkinter as tk
import time
from tkinter import messagebox

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from dotenv import load_dotenv

load_dotenv()

try:
    import winsound
except ImportError:
    winsound = None


class RecordingCancelled(Exception):
    pass


class RecordingDialog:
    def __init__(self, filename, prompt_text, fs=16000, channels=1):
        self.filename = filename
        self.prompt_text = prompt_text
        self.fs = fs
        self.channels = channels

        self.root = tk.Tk()
        self.root.title("Pronunciation Recorder")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.root.attributes("-topmost", True)

        self.status_var = tk.StringVar(value="録音前です。開始ボタンを押してください。")
        self.elapsed_var = tk.StringVar(value="00.0 sec")
        self.result = None

        self.stream = None
        self.frames = []
        self.audio_data = None
        self.is_recording = False
        self.playing = False
        self.record_started_at = None

        self._build_ui()

    def _build_ui(self):
        container = tk.Frame(self.root, padx=20, pady=16)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="発音する単語",
            font=("Yu Gothic UI", 10, "bold"),
        ).pack(anchor="w")

        tk.Label(
            container,
            text=self.prompt_text,
            font=("Yu Gothic UI", 18, "bold"),
            pady=6,
        ).pack(anchor="w")

        tk.Label(
            container,
            textvariable=self.status_var,
            justify="left",
            wraplength=360,
        ).pack(anchor="w", pady=(4, 2))

        tk.Label(
            container,
            text="開始 -> 停止 -> 再生で確認 -> 納得できなければ録り直し",
            fg="#666666",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            container,
            textvariable=self.elapsed_var,
            fg="#555555",
        ).pack(anchor="w", pady=(0, 12))

        row1 = tk.Frame(container)
        row1.pack(fill="x", pady=(0, 8))
        row2 = tk.Frame(container)
        row2.pack(fill="x")

        self.start_button = tk.Button(row1, text="録音開始", width=12, command=self.start_recording)
        self.stop_button = tk.Button(row1, text="停止", width=12, command=self.stop_recording, state="disabled")
        self.play_button = tk.Button(row1, text="再生", width=12, command=self.play_recording, state="disabled")

        self.retry_button = tk.Button(row2, text="録り直し", width=12, command=self.retry_recording, state="disabled")
        self.accept_button = tk.Button(row2, text="この音声で進む", width=16, command=self.accept_recording, state="disabled")
        self.cancel_button = tk.Button(row2, text="終了", width=12, command=self.cancel)

        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button.pack(side="left", padx=(0, 8))
        self.play_button.pack(side="left")
        self.retry_button.pack(side="left", padx=(0, 8))
        self.accept_button.pack(side="left", padx=(0, 8))
        self.cancel_button.pack(side="left")

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 3)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[WARN] Recording status: {status}")
        self.frames.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return

        self.stop_playback()
        self.frames = []
        self.audio_data = None

        try:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=self.channels,
                dtype="int16",
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            messagebox.showerror("録音開始エラー", str(exc), parent=self.root)
            return

        self.is_recording = True
        self.record_started_at = time.perf_counter()
        self.status_var.set("録音中です。話し終えたら停止を押してください。")
        self.elapsed_var.set("0.0 sec")

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.play_button.config(state="disabled")
        self.retry_button.config(state="disabled")
        self.accept_button.config(state="disabled")
        self._update_elapsed()

    def _update_elapsed(self):
        if not self.is_recording:
            return

        elapsed_sec = max(0.0, time.perf_counter() - float(self.record_started_at or time.perf_counter()))
        self.elapsed_var.set(f"{elapsed_sec:.1f} sec")
        self.root.after(100, self._update_elapsed)

    def stop_recording(self):
        if not self.is_recording:
            return

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
            self.is_recording = False

        if not self.frames:
            self.status_var.set("音声が取得できませんでした。もう一度録音してください。")
            self.elapsed_var.set("0.0 sec")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            return

        self.audio_data = np.concatenate(self.frames, axis=0)
        duration_sec = len(self.audio_data) / float(self.fs)
        self.status_var.set(f"録音を停止しました。長さ: {duration_sec:.1f} sec")
        self.elapsed_var.set(f"{duration_sec:.1f} sec")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.play_button.config(state="normal")
        self.retry_button.config(state="normal")
        self.accept_button.config(state="normal")

    def play_recording(self):
        if self.audio_data is None:
            return

        if winsound is None:
            messagebox.showinfo("再生不可", "この環境では WAV 再生を利用できません。", parent=self.root)
            return

        self.stop_playback()
        write(self.filename, self.fs, self.audio_data)
        winsound.PlaySound(self.filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.playing = True
        self.status_var.set("録音を再生中です。")

    def stop_playback(self):
        if winsound is not None and self.playing:
            winsound.PlaySound(None, 0)
        self.playing = False

    def retry_recording(self):
        self.stop_playback()
        self.audio_data = None
        self.frames = []
        self.status_var.set("録り直します。開始ボタンを押してください。")
        self.elapsed_var.set("0.0 sec")
        self.play_button.config(state="disabled")
        self.retry_button.config(state="disabled")
        self.accept_button.config(state="disabled")

    def accept_recording(self):
        if self.audio_data is None:
            return

        self.stop_playback()
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        write(self.filename, self.fs, self.audio_data)
        self.result = self.filename
        self.root.destroy()

    def cancel(self):
        if self.is_recording:
            self.stop_recording()
        self.stop_playback()
        self.result = None
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        if not self.result:
            raise RecordingCancelled("Recording was cancelled by the user.")
        return self.result


def record_audio(filename, duration=2, fs=16000):
    print(f"Recording: {filename}")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    write(filename, fs, audio)


def record_audio_with_gui(filename, prompt_text, fs=16000):
    dialog = RecordingDialog(filename=filename, prompt_text=prompt_text, fs=fs)
    return dialog.show()
