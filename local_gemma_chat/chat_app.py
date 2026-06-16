"""
Self-contained local Gemma 3 desktop chat.

No Ollama. The app downloads the Gemma 3 GGUF once from Hugging Face, then
auto-launches the bundled llama.cpp server (in ./llama_bin) in the background
on 127.0.0.1 and streams the conversation over localhost. After the first
download it runs 100% offline on CPU. Closing the window stops the engine.

Why a bundled server instead of in-process? The prebuilt llama-cpp-python wheel
requires AVX-512, which this CPU lacks. The official llama.cpp build ships
multiple CPU backends and auto-selects the right one (AVX2) at runtime.

Run:  .venv\Scripts\python.exe chat_app.py   (or double-click run.bat)
"""

import os
import json
import queue
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import scrolledtext

from huggingface_hub import hf_hub_download

# --- Config -----------------------------------------------------------------
# Switch the model size here. Bigger = smarter but slower on CPU + bigger download.
#   "1b"  ~0.8 GB  fast on CPU            (default, great for a desktop demo)
#   "4b"  ~3.3 GB  noticeably smarter, still CPU-usable
#   "12b" ~7-8 GB  needs a strong CPU / lots of RAM
#   "27b" ~16 GB   really wants a GPU
MODEL_SIZE = "4b"

_MODELS = {
    "1b": ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
    "4b": ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    "12b": ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf"),
    "27b": ("unsloth/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf"),
}

SYSTEM_PROMPT = "You are a helpful, friendly assistant running fully offline on the user's machine."
N_CTX = 4096                       # context window (conversation memory)
N_THREADS = os.cpu_count() or 4
HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_EXE = os.path.join(HERE, "llama_bin", "llama-server.exe")
CREATE_NO_WINDOW = 0x08000000      # don't pop a console window for the server


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Local Gemma 3 ({MODEL_SIZE}) — offline chat")
        self.root.geometry("720x620")
        self.root.configure(bg="#1e1e2e")

        self.proc = None
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.events = queue.Queue()   # background threads -> UI

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._drain_events)

        threading.Thread(target=self._start_engine, daemon=True).start()

    # --- UI ----------------------------------------------------------------
    def _build_ui(self):
        self.display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED,
            bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
            font=("Segoe UI", 11), borderwidth=0, padx=12, pady=12,
        )
        self.display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        self.display.tag_config("user", foreground="#89b4fa", font=("Segoe UI", 11, "bold"))
        self.display.tag_config("ai", foreground="#a6e3a1", font=("Segoe UI", 11, "bold"))
        self.display.tag_config("sys", foreground="#6c7086", font=("Segoe UI", 9, "italic"))
        self.display.tag_config("body", foreground="#cdd6f4")

        bar = tk.Frame(self.root, bg="#1e1e2e")
        bar.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.entry = tk.Entry(
            bar, font=("Segoe UI", 11), bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4", relief=tk.FLAT,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 6))
        self.entry.bind("<Return>", lambda e: self._on_send())

        self.send_btn = tk.Button(
            bar, text="Send", command=self._on_send, font=("Segoe UI", 10, "bold"),
            bg="#89b4fa", fg="#1e1e2e", relief=tk.FLAT, padx=18, activebackground="#74c7ec",
        )
        self.send_btn.pack(side=tk.RIGHT)

        self.status = tk.Label(
            self.root, text="Starting…", anchor=tk.W,
            bg="#11111b", fg="#f9e2af", font=("Segoe UI", 9), padx=10,
        )
        self.status.pack(fill=tk.X)

        self._set_input_enabled(False)
        self._append("sys", "", "Starting up — downloading/loading the model on first run…\n\n")

    def _set_input_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.entry.config(state=state)
        self.send_btn.config(state=state)
        if enabled:
            self.entry.focus_set()

    def _append(self, tag, label, text):
        self.display.config(state=tk.NORMAL)
        if label:
            self.display.insert(tk.END, label, tag)
        self.display.insert(tk.END, text, "body")
        self.display.see(tk.END)
        self.display.config(state=tk.DISABLED)

    # --- Engine startup (background thread) --------------------------------
    def _start_engine(self):
        try:
            repo, filename = _MODELS[MODEL_SIZE]
            self.events.put(("status", f"Downloading {filename} (first run only)…"))
            model_path = hf_hub_download(repo_id=repo, filename=filename)

            self.events.put(("status", "Starting local engine…"))
            log = open(os.path.join(HERE, "server.log"), "w", encoding="utf-8")
            self.proc = subprocess.Popen(
                [
                    SERVER_EXE,
                    "-m", model_path,
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "-c", str(N_CTX),
                    "-t", str(N_THREADS),
                    "-ngl", "0",          # CPU only
                ],
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )

            # Wait for the server's /health to report ready.
            deadline = time.time() + 180
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("engine exited early — see server.log")
                if self._health_ok():
                    self.events.put(("ready", None))
                    return
                time.sleep(0.5)
            raise TimeoutError("engine did not become ready in time")
        except Exception as e:
            self.events.put(("error", f"{type(e).__name__}: {e}"))

    def _health_ok(self):
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    # --- Sending / generation ----------------------------------------------
    def _on_send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._set_input_enabled(False)
        self.status.config(text="Thinking…")

        self.messages.append({"role": "user", "content": text})
        self._append("user", "You:  ", text + "\n")
        self._append("ai", "AI:   ", "")

        threading.Thread(target=self._generate, daemon=True).start()

    def _generate(self):
        try:
            body = json.dumps({
                "messages": self.messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 1024,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"},
            )
            reply = ""
            with urllib.request.urlopen(req) as resp:
                for raw in resp:                       # server-sent events, line by line
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if delta:
                        reply += delta
                        self.events.put(("token", delta))
            self.messages.append({"role": "assistant", "content": reply})
            self.events.put(("answer_done", None))
        except Exception as e:
            self.events.put(("error", f"{type(e).__name__}: {e}"))

    # --- Event pump (runs on UI thread) ------------------------------------
    def _drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                elif kind == "ready":
                    self._ready_status()
                    self._set_input_enabled(True)
                    self._append("sys", "", "Model ready. Ask me anything!\n\n")
                elif kind == "token":
                    self._append("body", "", payload)
                elif kind == "answer_done":
                    self._append("body", "", "\n\n")
                    self._ready_status()
                    self._set_input_enabled(True)
                elif kind == "error":
                    self._append("sys", "", f"\n[error] {payload}\n\n")
                    self.status.config(text="Error — see chat above / server.log")
                    self._set_input_enabled(True)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_events)

    def _ready_status(self):
        self.status.config(text=f"Ready · Gemma 3 {MODEL_SIZE} · CPU · {N_THREADS} threads · :{self.port}")

    # --- Shutdown -----------------------------------------------------------
    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()
