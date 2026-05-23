"""gui.py — Batch GUI for convert.py (BT.2020 PQ → Ultra HDR JPEG).

Usage:
    uv run python gui.py
"""

import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

PROJECT_DIR = Path(__file__).parent

_MSG_LOG        = "log"
_MSG_FILE_START = "file_start"   # payload: name
_MSG_FILE_DONE  = "file_done"    # payload: (done_1based, total, status, src_path)
_MSG_ALL_DONE   = "all_done"

_MAX_RETRIES = 1  # auto-retry each failed job this many times


# ── Worker ───────────────────────────────────────────────────────────────────

def _run_one(
    src: Path,
    out_dir: Path | None,
    opts: dict,
    q: queue.Queue,
    cancel: threading.Event,
) -> tuple[str, Path]:
    """Encode one file. Buffers all log text and emits it as a single block."""
    if cancel.is_set():
        q.put((_MSG_LOG, f"\n[skipped]  {src.name}\n"))
        return "skipped", src

    dst_dir = out_dir if out_dir else src.parent
    dst = dst_dir / f"{src.stem}.uhdr.jpg"
    q.put((_MSG_FILE_START, src.name))

    cmd = [
        sys.executable,
        str(PROJECT_DIR / "convert.py"),
        str(src), str(dst),
        "--quality",       str(opts["quality"]),
        "--gainmap-scale", str(opts["gainmap_scale"]),
        "--gainmap-gamma", str(opts["gainmap_gamma"]),
        "--peak-nits",     str(opts["peak_nits"]),
    ]
    if opts["force"]:
        cmd.append("--force")
    if opts["pipeline"] != "LUT":
        cmd += ["--pipeline", opts["pipeline"]]

    lines: list[str] = [f"\n── {src.name}  →  {dst.name} ──\n"]
    ok = False
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ok = result.returncode == 0
        if result.stderr:
            lines.append(result.stderr.rstrip() + "\n")
        if not ok and result.stdout:
            lines.append(result.stdout.rstrip() + "\n")
        if not ok:
            lines.append(f"  [exit {result.returncode}]\n")
    except Exception as exc:
        lines.append(f"  EXCEPTION: {exc}\n")

    q.put((_MSG_LOG, "".join(lines)))
    return ("ok" if ok else "error"), src


def _convert_worker(
    files: list[Path],
    out_dir: Path | None,
    opts: dict,
    q: queue.Queue,
    cancel: threading.Event,
    max_workers: int,
) -> None:
    total = len(files)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_run_one, src, out_dir, opts, q, cancel): src
                   for src in files}
        for fut in as_completed(futures):
            status, src = fut.result()
            done += 1
            q.put((_MSG_FILE_DONE, (done, total, status, src)))
    q.put((_MSG_ALL_DONE, None))


# ── GUI ──────────────────────────────────────────────────────────────────────

class App(TkinterDnD.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ultra HDR JPEG Batch Converter")
        self.minsize(700, 580)
        self._files: list[Path] = []
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._queue: queue.Queue = queue.Queue()
        self._after_id: str | None = None
        self._in_flight: int = 0
        self._done_count: int = 0
        self._success_count: int = 0
        self._failed_files: list[Path] = []
        self._retry_count: int = 0
        self._path_to_idx: dict[Path, int] = {}
        self._build_ui()
        self._update_convert_btn()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        PAD = dict(padx=8, pady=4)

        # ── Input files ──────────────────────────────────────────────────────
        frm_files = ttk.LabelFrame(self, text="Input TIFF files  (drag & drop accepted)")
        frm_files.pack(fill="both", expand=False, **PAD)

        btn_bar = ttk.Frame(frm_files)
        btn_bar.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Button(btn_bar, text="Add files…",      command=self._add_files).pack(side="left", padx=(0, 4))
        ttk.Button(btn_bar, text="Remove selected", command=self._remove_selected).pack(side="left", padx=(0, 4))
        ttk.Button(btn_bar, text="Clear all",       command=self._clear_files).pack(side="left")

        list_frm = ttk.Frame(frm_files)
        list_frm.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        vsb = ttk.Scrollbar(list_frm, orient="vertical")
        self._listbox = tk.Listbox(
            list_frm, selectmode="extended", height=6,
            yscrollcommand=vsb.set, activestyle="dotbox",
        )
        vsb.config(command=self._listbox.yview)
        vsb.pack(side="right", fill="y")
        self._listbox.pack(side="left", fill="both", expand=True)
        self._listbox.drop_target_register(DND_FILES)
        self._listbox.dnd_bind("<<Drop>>", self._on_drop)

        # ── Output folder ────────────────────────────────────────────────────
        frm_out = ttk.LabelFrame(self, text="Output folder  (blank = same folder as each input)")
        frm_out.pack(fill="x", **PAD)

        out_row = ttk.Frame(frm_out)
        out_row.pack(fill="x", padx=4, pady=4)
        self._out_var = tk.StringVar()
        ttk.Entry(out_row, textvariable=self._out_var).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(out_row, text="Browse…", command=self._browse_out).pack(side="left", padx=(0, 4))
        ttk.Button(out_row, text="Clear",   command=lambda: self._out_var.set("")).pack(side="left")

        # ── Encoder options ──────────────────────────────────────────────────
        frm_opts = ttk.LabelFrame(self, text="Encoder options")
        frm_opts.pack(fill="x", **PAD)

        g = ttk.Frame(frm_opts)
        g.pack(padx=8, pady=6, anchor="w")

        self._quality_var     = tk.IntVar(value=95)
        self._gm_scale_var    = tk.IntVar(value=1)
        self._gm_gamma_var    = tk.StringVar(value="1.0")
        self._peak_nits_var   = tk.StringVar(value="1000")
        self._jobs_var        = tk.IntVar(value=2)
        self._force_var       = tk.BooleanVar(value=False)
        self._pipeline_var = tk.StringVar(value="LUT")

        fields = [
            ("Quality (0–100):",       self._quality_var,   "spin",  0,    100),
            ("Gainmap scale (1–128):", self._gm_scale_var,  "spin",  1,    128),
            ("Gainmap gamma:",         self._gm_gamma_var,  "entry", None, None),
            ("Peak nits (203–10000):", self._peak_nits_var, "entry", None, None),
            ("Parallel jobs (1–8):",   self._jobs_var,      "spin",  1,    8),
        ]
        for n, (label, var, kind, lo, hi) in enumerate(fields):
            col = (n % 2) * 3
            row = n // 2
            ttk.Label(g, text=label, anchor="e").grid(row=row, column=col, sticky="e", padx=(8, 4))
            if kind == "spin":
                w: ttk.Widget = ttk.Spinbox(g, from_=lo, to=hi, textvariable=var, width=8)
            else:
                w = ttk.Entry(g, textvariable=var, width=10)
            w.grid(row=row, column=col + 1, sticky="w", padx=(0, 24), pady=2)

        ttk.Label(g, text="Pipeline:", anchor="e").grid(
            row=2, column=3, sticky="e", padx=(8, 4)
        )
        ttk.Combobox(
            g, textvariable=self._pipeline_var,
            values=["LUT", "Parametric"], state="readonly", width=11,
        ).grid(row=2, column=4, sticky="w", padx=(0, 24), pady=2)

        ttk.Checkbutton(g, text="Force overwrite  (--force)", variable=self._force_var).grid(
            row=3, column=0, columnspan=5, sticky="w", padx=(8, 0), pady=(4, 0)
        )

        # ── Convert / Cancel button ──────────────────────────────────────────
        self._convert_btn = ttk.Button(self, text="Convert", command=self._start, width=28)
        self._convert_btn.pack(pady=6)

        # ── Progress ─────────────────────────────────────────────────────────
        frm_prog = ttk.LabelFrame(self, text="Progress")
        frm_prog.pack(fill="x", **PAD)

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm_prog, textvariable=self._status_var).pack(anchor="w", padx=6, pady=(4, 0))

        # indeterminate bar — animates while any file is in flight
        self._cur_bar = ttk.Progressbar(frm_prog, mode="determinate", maximum=100, value=0)
        self._cur_bar.pack(fill="x", padx=6, pady=(2, 0))

        # overall completed-files bar
        overall_row = ttk.Frame(frm_prog)
        overall_row.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Label(overall_row, text="Overall:").pack(side="left", padx=(0, 4))
        self._overall_bar = ttk.Progressbar(overall_row, mode="determinate", maximum=1, value=0)
        self._overall_bar.pack(side="left", fill="x", expand=True)
        self._overall_label = ttk.Label(overall_row, text="0 / 0", width=8, anchor="e")
        self._overall_label.pack(side="left", padx=(4, 0))

        # ── Log ──────────────────────────────────────────────────────────────
        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, **PAD)

        _mono_font = ("Menlo", 10) if sys.platform == "darwin" else ("Consolas", 9)
        self._log = scrolledtext.ScrolledText(
            frm_log, height=8, state="disabled",
            font=_mono_font, wrap="word",
        )
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

    # ── File-list helpers ────────────────────────────────────────────────────

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select BT.2020 PQ TIFF files",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        existing = {str(p) for p in self._files}
        for p in paths:
            if p not in existing:
                self._files.append(Path(p))
                self._listbox.insert(tk.END, Path(p).name)
        self._update_convert_btn()

    def _remove_selected(self) -> None:
        for idx in reversed(self._listbox.curselection()):
            self._listbox.delete(idx)
            del self._files[idx]
        self._update_convert_btn()

    def _clear_files(self) -> None:
        self._listbox.delete(0, tk.END)
        self._files.clear()
        self._update_convert_btn()

    def _browse_out(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self._out_var.set(d)

    def _on_drop(self, event) -> None:
        existing = {str(p) for p in self._files}
        for raw in self.tk.splitlist(event.data):
            path = Path(raw)
            if path.suffix.lower() in (".tif", ".tiff") and str(path) not in existing:
                self._files.append(path)
                self._listbox.insert(tk.END, path.name)
                existing.add(str(path))
        self._update_convert_btn()

    def _update_convert_btn(self) -> None:
        n = len(self._files)
        if n == 0:
            self._convert_btn.config(text="Convert", state="disabled")
        else:
            self._convert_btn.config(
                text=f"Convert  {n} file{'s' if n != 1 else ''}",
                state="normal",
            )

    # ── Conversion control ───────────────────────────────────────────────────

    def _collect_opts(self) -> dict:
        try:
            q  = int(self._quality_var.get())
            gs = int(self._gm_scale_var.get())
            gg = float(self._gm_gamma_var.get())
            pn = float(self._peak_nits_var.get())
            jb = int(self._jobs_var.get())
        except (ValueError, tk.TclError) as exc:
            raise ValueError(f"Invalid option value: {exc}") from exc
        return {
            "quality":       max(0,     min(100,   q)),
            "gainmap_scale": max(1,     min(128,   gs)),
            "gainmap_gamma": max(0.001,            gg),
            "peak_nits":     max(203.0, min(10000, pn)),
            "jobs":          max(1,     min(8,     jb)),
            "force":         self._force_var.get(),
            "pipeline":      self._pipeline_var.get().lower(),
        }

    def _start(self) -> None:
        if not self._files:
            return
        try:
            opts = self._collect_opts()
        except ValueError as exc:
            messagebox.showerror("Invalid options", str(exc), parent=self)
            return

        out_str = self._out_var.get().strip()
        out_dir = Path(out_str) if out_str else None
        if out_dir and not out_dir.is_dir():
            messagebox.showerror("Output folder", f"Folder does not exist:\n{out_dir}", parent=self)
            return

        n = len(self._files)
        self._overall_bar.config(maximum=n, value=0)
        self._overall_label.config(text=f"0 / {n}")
        self._cur_bar.config(mode="indeterminate", value=0)
        self._cur_bar.start(12)
        self._status_var.set(f"Starting  ({opts['jobs']} parallel job{'s' if opts['jobs'] != 1 else ''})…")
        self._in_flight = 0
        self._done_count = 0
        self._success_count = 0
        self._failed_files = []
        self._retry_count = 0
        self._path_to_idx = {p: i for i, p in enumerate(self._files)}
        for i in range(len(self._files)):
            self._listbox.itemconfig(i, bg="", fg="")
        self._log_clear()
        self._convert_btn.config(text="Cancel", command=self._cancel, state="normal")
        self._cancel_event.clear()

        self._thread = threading.Thread(
            target=_convert_worker,
            args=(list(self._files), out_dir, opts, self._queue,
                  self._cancel_event, opts["jobs"]),
            daemon=True,
        )
        self._thread.start()
        self._after_id = self.after(80, self._poll)

    def _cancel(self) -> None:
        self._cancel_event.set()
        self._convert_btn.config(text="Cancelling…", state="disabled")

    def _poll(self) -> None:
        total = len(self._files)
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == _MSG_LOG:
                    self._log_append(payload)
                elif kind == _MSG_FILE_START:
                    self._in_flight += 1
                    self._status_var.set(
                        f"  {self._in_flight} in flight  |  "
                        f"{self._done_count} / {total} done"
                    )
                elif kind == _MSG_FILE_DONE:
                    done, tot, status, src = payload
                    self._in_flight = max(0, self._in_flight - 1)
                    self._done_count = done
                    self._overall_bar["value"] = done
                    idx = self._path_to_idx.get(src)
                    if status == "ok":
                        self._success_count += 1
                        mark = "✓"
                        if idx is not None:
                            self._listbox.itemconfig(idx, bg="#d4edda", fg="#155724")
                    elif status == "error":
                        self._failed_files.append(src)
                        mark = "✗"
                        if idx is not None:
                            self._listbox.itemconfig(idx, bg="#f8d7da", fg="#721c24")
                    else:
                        mark = "—"
                    n_err = len(self._failed_files)
                    label = f"{self._success_count}✓" + (f"  {n_err}✗" if n_err else "") + f"  / {tot}"
                    self._overall_label.config(text=label)
                    self._status_var.set(
                        f"  {mark} {src.name}  |  "
                        f"{self._in_flight} in flight  |  {done} / {tot} done"
                    )
                elif kind == _MSG_ALL_DONE:
                    self._on_done()
                    return
        except queue.Empty:
            pass
        self._after_id = self.after(80, self._poll)

    def _on_done(self) -> None:
        self._cur_bar.stop()
        self._cur_bar.config(mode="determinate", value=100)

        if self._failed_files and self._retry_count < _MAX_RETRIES and not self._cancel_event.is_set():
            self._start_retry()
            return

        n_total = len(self._files)
        n_fail = len(self._failed_files)
        n_ok = self._success_count
        if n_fail == 0:
            msg = f"Done — {n_ok} file{'s' if n_ok != 1 else ''} converted."
        else:
            retry_note = f" (retried {_MAX_RETRIES}×)" if self._retry_count else ""
            msg = f"Done — {n_ok} succeeded,  {n_fail} failed{retry_note}."
        self._status_var.set(msg)
        self._convert_btn.config(
            text=f"Convert  {n_total} file{'s' if n_total != 1 else ''}",
            command=self._start,
            state="normal",
        )
        self._after_id = None

    def _start_retry(self) -> None:
        self._retry_count += 1
        files_to_retry = list(self._failed_files)
        self._failed_files = []
        self._success_count = 0
        n = len(files_to_retry)

        for src in files_to_retry:
            idx = self._path_to_idx.get(src)
            if idx is not None:
                self._listbox.itemconfig(idx, bg="#fff3cd", fg="#856404")

        out_str = self._out_var.get().strip()
        out_dir = Path(out_str) if out_str else None
        opts = self._collect_opts()
        jobs = opts["jobs"]

        self._overall_bar.config(maximum=n, value=0)
        self._overall_label.config(text=f"0 / {n}")
        self._cur_bar.config(mode="indeterminate", value=0)
        self._cur_bar.start(12)
        self._status_var.set(f"Retrying {n} failed file{'s' if n != 1 else ''}…")
        self._in_flight = 0
        self._done_count = 0
        self._log_append(f"\n── Retry {self._retry_count} — {n} file{'s' if n != 1 else ''} ──\n")
        self._cancel_event.clear()
        self._convert_btn.config(text="Cancel", command=self._cancel, state="normal")

        self._thread = threading.Thread(
            target=_convert_worker,
            args=(files_to_retry, out_dir, opts, self._queue,
                  self._cancel_event, jobs),
            daemon=True,
        )
        self._thread.start()
        self._after_id = self.after(80, self._poll)

    # ── Log helpers ──────────────────────────────────────────────────────────

    def _log_append(self, text: str) -> None:
        self._log.config(state="normal")
        self._log.insert(tk.END, text)
        self._log.see(tk.END)
        self._log.config(state="disabled")

    def _log_clear(self) -> None:
        self._log.config(state="normal")
        self._log.delete("1.0", tk.END)
        self._log.config(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
