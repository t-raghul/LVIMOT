from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import json
import math
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from lvimot.output.pipeline_inspector import PIPELINE_STAGES
from lvimot.inputs.sensor_profile import with_lidar_profile
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


APP_TITLE = "LVIMOT Live Dashboard"
DEFAULT_CARLA_INSTALL = os.environ.get("CARLA_INSTALL", "")
DEFAULT_CALIBRATION = str(Path(__file__).resolve().parents[2] / 'config' / 'carla_fast_mapping.json')
DEFAULT_OUTPUT_ROOT = "outputs_carla"
FRAME_RE = re.compile(
    r"frame\s+(?P<frame>\d+)\s+/\s+CARLA\s+(?P<carla>\d+)\s+\|\s+"
    r"(?P<ms>[0-9.]+)\s+ms\s+\|\s+RSS\s+(?P<rss>[0-9.]+)\s+MB\s+\|\s+"
    r"tracks\s+(?P<tracks>\d+)\s+\|\s+map\s+(?P<map>\d+)\s+\|\s+"
    r"stationary=(?P<stationary>True|False)\s+\|\s+sync=(?P<sync>[0-9.]+)\s+ms"
    r"(?:\s+\|\s+camlag=(?P<camlag>\d+)f)?"
    r"(?:\s+\|\s+pump=(?P<pump>\d+))?"
)


PRESETS = {
    "Validated": {
        "width": 1242,
        "height": 375,
        "fov": 90.0,
        "yolo_size": 1280,
        "confidence": 0.06,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
    "High Recall": {
        "width": 1242,
        "height": 375,
        "fov": 90.0,
        "yolo_size": 1280,
        "confidence": 0.035,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
    "Fast": {
        "width": 828,
        "height": 250,
        "fov": 90.0,
        "yolo_size": 640,
        "confidence": 0.08,
        "vehicles": 8,
        "fixed_delta": 0.10,
    },
    "Balanced": {
        "width": 1024,
        "height": 309,
        "fov": 90.0,
        "yolo_size": 960,
        "confidence": 0.06,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
    "Quality": {
        "width": 1656,
        "height": 500,
        "fov": 90.0,
        "yolo_size": 1280,
        "confidence": 0.05,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
}


class LVIMOTLiveGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        # Fit the dashboard to the actual desktop instead of assuming a tall
        # 900 px client area.  The control column is independently scrollable
        # and the START/STOP footer is always pinned on-screen.
        screen_w = max(1024, int(self.root.winfo_screenwidth()))
        screen_h = max(700, int(self.root.winfo_screenheight()))
        # Reserve a genuine third column for the inspector; use a wider
        # window on large desktops instead of stealing all live-camera space.
        window_w = min(1790, max(1160, screen_w - 48))
        window_h = min(960, max(650, screen_h - 96))
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(1120, 650)

        self.project_root = Path(__file__).resolve().parents[2]
        self.python_exe = self.project_root / ".venv_live38" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not self.python_exe.is_file():
            self.python_exe = Path(sys.executable)
        self.main_script = self.project_root / 'lvimot/output/cli.py'
        self.config_path = self.project_root / "lvimot_gui_config.json"

        self.backend_process: Optional[subprocess.Popen] = None
        self.carla_process: Optional[subprocess.Popen] = None
        self.stop_requested = False
        self.events: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.latest_record: Dict = {}
        self.pose_history: List[Tuple[float, float]] = []
        self.map_history: List[Tuple[int, int]] = []
        self.track_history: Dict[int, Dict] = {}
        self.spatial_snapshot: Dict = {}
        self._spatial_snapshot_mtime_ns: Optional[int] = None
        self._camera_snapshot_mtime_ns: Optional[int] = None
        self._camera_photo = None
        self.max_tracks_seen = 0
        self.processed_records = 0
        self.last_record_frame: Optional[int] = None
        self.log_file_offset = 0
        self.log_file_identity: Optional[Tuple[int, int]] = None
        self._control_widgets: List[tk.Widget] = []
        self.inspector_latest_snapshot: Dict = {}
        self.inspector_display_snapshot: Dict = {}
        self.inspector_selected_stage = "carla"
        self.inspector_auto_index = -1
        self._inspector_stage_items: Dict[str, Tuple[int, int]] = {}

        self._build_variables()
        self._build_ui()
        self._apply_preset("Fast")
        self._load_saved_config(silent=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_events)
        self.root.after(800, self._poll_carla_status)
        self.root.after(500, self._redraw_bev)
        self.root.after(350, self._poll_spatial_map_snapshot)
        self.root.after(400, self._poll_camera_snapshot)
        self.root.after(550, self._redraw_map_health)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def _build_variables(self):
        self.var_carla_install = tk.StringVar(value=DEFAULT_CARLA_INSTALL)
        self.var_calibration = tk.StringVar(value=DEFAULT_CALIBRATION)
        self.var_host = tk.StringVar(value="127.0.0.1")
        self.var_port = tk.StringVar(value="2000")
        self.var_render_quality = tk.StringVar(value="Epic")
        self.var_show_carla_window = tk.BooleanVar(value=False)

        self.var_preset = tk.StringVar(value="Fast")
        self.var_autopilot = tk.BooleanVar(value=True)
        self.var_gt_eval = tk.BooleanVar(value=True)
        # Dashboard camera/map streaming is always enabled in GUI mode so the
        # live ego RGB view remains visible regardless of the selected tab.
        self.var_processed_display = tk.BooleanVar(value=True)
        self.var_show_trajectory = tk.BooleanVar(value=False)
        self.var_show_object_tracks = tk.BooleanVar(value=True)
        self.var_candidate_hold = tk.StringVar(value="4")
        self.var_vehicles = tk.StringVar(value="8")
        self.var_duration = tk.StringVar(value="60")
        self.var_fixed_delta = tk.StringVar(value="0.10")
        self.var_seed = tk.StringVar(value="42")
        self.var_sensor_timeout = tk.StringVar(value="10")
        self.var_output_root = tk.StringVar(value=DEFAULT_OUTPUT_ROOT)
        self.var_sliding_window_size = tk.StringVar(value=str(DEFAULT_LIMITS.sliding_window_size))
        self.var_map_voxel_size = tk.StringVar(value=str(DEFAULT_LIMITS.map_voxel_size_m))
        self.var_map_radius = tk.StringVar(value=str(DEFAULT_LIMITS.map_rolling_radius_m))
        self.var_map_max_in_memory = tk.StringVar(value=str(DEFAULT_LIMITS.max_map_static_voxels))

        self.var_width = tk.StringVar(value="828")
        self.var_height = tk.StringVar(value="250")
        self.var_fov = tk.StringVar(value="90.0")
        self.var_yolo_model = tk.StringVar(value="models/yolov8n.pt")
        self.var_yolo_size = tk.StringVar(value="640")
        self.var_confidence = tk.StringVar(value="0.08")
        self.var_yolo_device = tk.StringVar(value="auto")

        self.var_carla_status = tk.StringVar(value="OFFLINE")
        self.var_backend_status = tk.StringVar(value="STOPPED")
        self.var_frame = tk.StringVar(value="-")
        self.var_carla_frame = tk.StringVar(value="-")
        self.var_total_ms = tk.StringVar(value="-")
        self.var_fps = tk.StringVar(value="-")
        self.var_rss = tk.StringVar(value="-")
        self.var_tracks = tk.StringVar(value="0")
        self.var_map = tk.StringVar(value="0")
        self.var_stationary = tk.StringVar(value="-")
        self.var_sync = tk.StringVar(value="-")
        self.var_camlag = tk.StringVar(value="-")
        self.var_pump = tk.StringVar(value="-")
        self.var_pose = tk.StringVar(value="x=-  y=-  z=-  yaw=-")
        self.var_speed = tk.StringVar(value="-")
        self.var_sensor = tk.StringVar(value="Waiting for backend")
        self.var_tracks_note = tk.StringVar(value="No active published tracks yet.")
        self.var_map_note = tk.StringVar(value="4-D spatial map waiting for live data.")
        self.var_camera_note = tk.StringVar(value="Waiting for the first live RGB frame. The camera stays visible across tabs.")
        self.var_health = tk.StringVar(value="WAITING")
        self.var_health_detail = tk.StringVar(value="Start LVIMOT to evaluate live system health.")
        self.var_last_error = tk.StringVar(value="")
        self.var_map_view = tk.StringVar(value="Clean 2D")
        self.var_map_zoom = tk.DoubleVar(value=30.0)
        self.var_map_angle = tk.DoubleVar(value=35.0)
        self.var_map_view_note = tk.StringVar(value="Clean 2D | filtering affects display only")
        self.var_inspector_auto = tk.BooleanVar(value=False)
        self.var_inspector_freeze = tk.BooleanVar(value=False)
        self.var_inspector_frame = tk.StringVar(value="No completed frame yet")
        self.var_inspector_mode = tk.StringVar(value="Click any box to inspect an algorithm and its latest measurements.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)

        # Left side: fixed-width shell.  Settings scroll in the middle while
        # the run controls stay pinned at the bottom and can never disappear.
        left_shell = ttk.Frame(self.root, padding=(8, 8, 6, 8), width=350)
        right = ttk.Frame(self.root, padding=(0, 8, 4, 8))
        inspector_shell = ttk.Frame(self.root, padding=(2, 8, 8, 8), width=358)
        left_shell.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")
        inspector_shell.grid(row=0, column=2, sticky="nsew")
        left_shell.grid_propagate(False)
        inspector_shell.grid_propagate(False)
        left_shell.columnconfigure(0, weight=1)
        left_shell.rowconfigure(1, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=2)
        right.rowconfigure(2, weight=3)

        title = ttk.Label(
            left_shell,
            text="LVIMOT LIVE CONTROLS",
            font=("Segoe UI", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        scroll_host = ttk.Frame(left_shell)
        scroll_host.grid(row=1, column=0, sticky="nsew")
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)

        self.control_canvas = tk.Canvas(
            scroll_host,
            borderwidth=0,
            highlightthickness=0,
            background=self.root.cget("background"),
        )
        control_scrollbar = ttk.Scrollbar(
            scroll_host,
            orient="vertical",
            command=self.control_canvas.yview,
        )
        self.control_canvas.configure(yscrollcommand=control_scrollbar.set)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar.grid(row=0, column=1, sticky="ns")

        self.control_content = ttk.Frame(self.control_canvas, padding=(0, 0, 4, 0))
        self._control_window = self.control_canvas.create_window(
            (0, 0),
            window=self.control_content,
            anchor="nw",
        )
        self.control_content.bind("<Configure>", self._on_control_content_configure)
        self.control_canvas.bind("<Configure>", self._on_control_canvas_configure)

        # Mouse wheel works anywhere over the settings column, including over
        # entries and labels.  The handler checks pointer position so the BEV,
        # console and track table keep their own scrolling behavior.
        self.root.bind_all("<MouseWheel>", self._on_control_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_control_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_control_mousewheel, add="+")

        self._build_controls(self.control_content)
        self._build_run_footer(left_shell)
        self._build_status_cards(right)
        self._build_dashboard(right)
        self._build_inspector(inspector_shell)

    def _on_control_content_configure(self, _event=None):
        try:
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _on_control_canvas_configure(self, event):
        try:
            self.control_canvas.itemconfigure(self._control_window, width=max(1, event.width))
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _pointer_over_control_canvas(self, event) -> bool:
        try:
            x = int(getattr(event, "x_root", self.root.winfo_pointerx()))
            y = int(getattr(event, "y_root", self.root.winfo_pointery()))
            left = self.control_canvas.winfo_rootx()
            top = self.control_canvas.winfo_rooty()
            right = left + self.control_canvas.winfo_width()
            bottom = top + self.control_canvas.winfo_height()
            return left <= x <= right and top <= y <= bottom
        except tk.TclError:
            return False

    def _on_control_mousewheel(self, event):
        if not self._pointer_over_control_canvas(event):
            return

        try:
            first, last = self.control_canvas.yview()
        except tk.TclError:
            return
        if first <= 0.0 and last >= 1.0:
            return

        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return
            direction = -1 if delta > 0 else 1

        self.control_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _build_controls(self, parent):
        server = ttk.LabelFrame(parent, text="CARLA Server", padding=8)
        server.pack(fill="x", pady=(0, 4))
        self._entry_row(server, "Install", self.var_carla_install, browse=self._browse_carla)
        self._entry_row(server, "Host", self.var_host)
        self._entry_row(server, "Port", self.var_port)
        self._combo_row(server, "Render quality", self.var_render_quality, ["Low", "Epic"])
        self._check_row(server,'Show CARLA spectator window',self.var_show_carla_window)

        server_buttons = ttk.Frame(server)
        server_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(server_buttons, text="Check CARLA", command=self._check_carla_now).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(server_buttons, text="Start CARLA", command=self._start_carla).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(server_buttons, text="Stop CARLA", command=self._stop_carla).pack(side="left", expand=True, fill="x", padx=(3, 0))

        run = ttk.LabelFrame(parent, text="Scenario", padding=8)
        run.pack(fill="x", pady=4)
        self._combo_row(run, "Preset", self.var_preset, list(PRESETS.keys()), callback=self._preset_changed)
        self._check_row(run, "Ego autopilot", self.var_autopilot)
        self._check_row(run, "GT evaluation only", self.var_gt_eval)
        always_stream = ttk.Checkbutton(
            run, text="Live camera + spatial 4-D map stream (always on)",
            variable=self.var_processed_display, state="disabled"
        )
        always_stream.pack(anchor="w", pady=2)
        self._check_row(run, "Show estimated ego trail in Overview", self.var_show_trajectory)
        self._entry_row(run, "NPC vehicles", self.var_vehicles)
        self._entry_row(run, "Run seconds (real time)", self.var_duration)
        self._entry_row(run, "Random seed", self.var_seed)

        perception = ttk.LabelFrame(parent, text="Camera / Perception", padding=8)
        perception.pack(fill="x", pady=4)
        self._entry_row(perception, "Calibration", self.var_calibration, browse=self._browse_calibration)
        self._entry_row(perception, "Width", self.var_width)
        self._entry_row(perception, "Height", self.var_height)
        self._entry_row(perception, "Horizontal FOV", self.var_fov)
        self._entry_row(perception, "YOLO model", self.var_yolo_model)
        self._combo_row(perception, "YOLO imgsz", self.var_yolo_size, ["640", "832", "960", "1280"])
        self._entry_row(perception, "Confidence", self.var_confidence)
        self._combo_row(perception, "Candidate hold", self.var_candidate_hold, ["0", "2", "3", "4", "5", "6", "8"])
        self._combo_row(perception, "YOLO device", self.var_yolo_device, ["auto", "0", "cpu"])

        advanced = ttk.LabelFrame(parent, text="Advanced", padding=8)
        advanced.pack(fill="x", pady=4)
        self._combo_row(advanced, "Fixed delta", self.var_fixed_delta, ["0.05", "0.10", "0.20"])
        self._entry_row(advanced, "Sensor timeout", self.var_sensor_timeout)
        self._entry_row(advanced, "Output root", self.var_output_root)
        self._entry_row(advanced, "Sliding window", self.var_sliding_window_size)
        self._entry_row(advanced, "Voxel size (m)", self.var_map_voxel_size)
        self._entry_row(advanced, "Map radius (m)", self.var_map_radius)
        self._entry_row(advanced, "Max in memory", self.var_map_max_in_memory)

        note = ttk.Label(
            parent,
            text=(
                "Validated baseline: 1242×375, FOV 90°, YOLOv8n, imgsz 1280, "
                "conf 0.06, 10 Hz.\nThe annotated live ego camera is permanently visible in the dashboard. "
                "Candidate hold is display-only continuity for brief fused-detection gaps; it never creates a MOT track. "
                ""
                "Scroll this panel for settings; START/STOP stays pinned below."
            ),
            wraplength=325,
            justify="left",
        )
        note.pack(fill="x", pady=(6, 8))

    def _build_run_footer(self, parent):
        footer = ttk.Frame(parent)
        footer.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        footer.columnconfigure(0, weight=1)

        launch = ttk.LabelFrame(footer, text="Run", padding=8)
        launch.grid(row=0, column=0, sticky="ew")

        row1 = ttk.Frame(launch)
        row1.pack(fill="x")
        self.btn_start = ttk.Button(
            row1,
            text="START LVIMOT",
            command=self._start_backend,
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_stop = ttk.Button(
            row1,
            text="STOP",
            command=self._stop_backend,
            state="disabled",
        )
        self.btn_stop.pack(side="left", expand=True, fill="x")

        row2 = ttk.Frame(launch)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Button(row2, text="Save", command=self._save_config).pack(
            side="left", expand=True, fill="x", padx=(0, 3)
        )
        ttk.Button(row2, text="Load", command=self._load_saved_config).pack(
            side="left", expand=True, fill="x", padx=3
        )
        ttk.Button(row2, text="Outputs", command=self._open_outputs).pack(
            side="left", expand=True, fill="x", padx=(3, 0)
        )

        ttk.Label(
            footer,
            textvariable=self.var_last_error,
            foreground="#b00020",
            wraplength=335,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    def _build_status_cards(self, parent):
        """Compact two-line status strip to preserve space for the live camera."""
        strip = ttk.Frame(parent)
        strip.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for c in range(7):
            strip.columnconfigure(c, weight=1)

        cards = [
            ("CARLA", self.var_carla_status),
            ("Backend", self.var_backend_status),
            ("Frame", self.var_frame),
            ("Pipe", self.var_total_ms),
            ("FPS", self.var_fps),
            ("Tracks", self.var_tracks),
            ("Map", self.var_map),
            ("RSS", self.var_rss),
            ("Move", self.var_stationary),
            ("Sync", self.var_sync),
            ("Cam lag", self.var_camlag),
            ("Pump", self.var_pump),
            ("Speed", self.var_speed),
        ]
        for i, (name, variable) in enumerate(cards):
            row = i // 7
            col = i % 7
            cell = ttk.Frame(strip, padding=(3, 1))
            cell.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 2, 0), pady=(0 if row == 0 else 2, 0))
            ttk.Label(cell, text=name, font=("Segoe UI", 8)).pack()
            ttk.Label(cell, textvariable=variable, font=("Segoe UI", 9, "bold")).pack()

    def _build_dashboard(self, parent):
        # The camera is intentionally NOT a tab.  It is always visible while the
        # backend runs; the notebook below is for secondary views/diagnostics.
        camera_shell = ttk.LabelFrame(parent, text="Live Ego RGB", padding=(5, 4))
        camera_shell.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        camera_shell.columnconfigure(0, weight=1)
        camera_shell.rowconfigure(1, weight=1)
        ttk.Label(
            camera_shell,
            textvariable=self.var_camera_note,
            wraplength=1100,
            justify="left",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 3))
        self.camera_canvas = tk.Canvas(camera_shell, background="#0b1117", highlightthickness=0)
        self.camera_canvas.grid(row=1, column=0, sticky="nsew")

        notebook = ttk.Notebook(parent)
        notebook.grid(row=2, column=0, sticky="nsew")

        overview = ttk.Frame(notebook, padding=6)
        tracks = ttk.Frame(notebook, padding=6)
        map_health = ttk.Frame(notebook, padding=6)
        console = ttk.Frame(notebook, padding=6)
        settings = ttk.Frame(notebook, padding=6)
        notebook.add(overview, text="Overview / BEV")
        notebook.add(tracks, text="Tracks")
        notebook.add(map_health, text="4-D Map / Health")
        notebook.add(console, text="Console")
        notebook.add(settings, text="Current Command")

        overview.columnconfigure(0, weight=1)
        overview.rowconfigure(1, weight=1)
        info = ttk.Frame(overview)
        info.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        ttk.Label(info, textvariable=self.var_pose, font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.var_sensor, font=("Consolas", 9)).grid(row=0, column=1, sticky="e")
        self.bev_canvas = tk.Canvas(overview, background="#111820", highlightthickness=0)
        self.bev_canvas.grid(row=1, column=0, sticky="nsew")

        tracks.columnconfigure(0, weight=1)
        tracks.rowconfigure(1, weight=1)
        ttk.Label(
            tracks,
            textvariable=self.var_tracks_note,
            wraplength=1000,
            justify="left",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        columns = ("id", "status", "class", "distance", "speed", "x", "y", "state", "hits", "last_seen")
        self.track_table = ttk.Treeview(tracks, columns=columns, show="headings")
        headings = {
            "id": "ID", "status": "Status", "class": "Class",
            "distance": "Distance m", "speed": "Speed m/s",
            "x": "X", "y": "Y", "state": "Motion",
            "hits": "Hits", "last_seen": "Last seen",
        }
        widths = {
            "id": 50, "status": 78, "class": 95, "distance": 90, "speed": 85,
            "x": 75, "y": 75, "state": 95, "hits": 60, "last_seen": 80,
        }
        for col in columns:
            self.track_table.heading(col, text=headings[col])
            self.track_table.column(col, width=widths[col], anchor="center")
        scroll = ttk.Scrollbar(tracks, orient="vertical", command=self.track_table.yview)
        self.track_table.configure(yscrollcommand=scroll.set)
        self.track_table.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

        map_health.columnconfigure(0, weight=1)
        map_health.rowconfigure(3, weight=1)
        health_header = ttk.Frame(map_health)
        health_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        health_header.columnconfigure(1, weight=1)
        ttk.Label(health_header, text="Live system health:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(health_header, textvariable=self.var_health, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(
            map_health, textvariable=self.var_health_detail, wraplength=1000,
            justify="left", font=("Segoe UI", 8),
        ).grid(row=1, column=0, sticky="ew", pady=(0, 4))
        toolbar = ttk.Frame(map_health)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(0, weight=1)
        toolbar.columnconfigure(1, weight=0)
        first_row = ttk.Frame(toolbar)
        first_row.grid(row=0, column=0, sticky="w")
        ttk.Label(first_row, text="Map view:").pack(side="left", padx=(0, 4))
        selector = ttk.Combobox(
            first_row, textvariable=self.var_map_view, state="readonly", width=17,
            values=("Clean 2D", "Connected lines", "3D height view", "Measured dots"),
        )
        selector.pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Save Map (.PLY)", command=self._save_map_ply).grid(
            row=0, column=1, sticky="e")
        second_row = ttk.Frame(toolbar)
        second_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Label(second_row, text="Range:").pack(side="left", padx=(0, 2))
        ttk.Scale(second_row, from_=15, to=60, variable=self.var_map_zoom, length=110).pack(side="left")
        ttk.Label(second_row, text="Rotate 3D:").pack(side="left", padx=(10, 2))
        ttk.Scale(second_row, from_=-80, to=80, variable=self.var_map_angle, length=105).pack(side="left")
        ttk.Checkbutton(second_row, text="Object IDs / paths / 3D boxes",
                        variable=self.var_show_object_tracks).pack(side="left", padx=(12, 0))

        self.map_canvas = tk.Canvas(map_health, background="#111820", highlightthickness=0)
        self.map_canvas.grid(row=3, column=0, sticky="nsew")
        self.map_canvas.bind("<B1-Motion>", self._rotate_map_3d)
        ttk.Label(
            map_health, textvariable=self.var_map_note, wraplength=1000,
            justify="left", font=("Segoe UI", 8),
        ).grid(row=4, column=0, sticky="ew", pady=(4, 0))

        console.columnconfigure(0, weight=1)
        console.rowconfigure(0, weight=1)
        self.console = tk.Text(console, wrap="none", background="#0d1117", foreground="#d9e1e8", insertbackground="white", font=("Consolas", 8))
        cscroll_y = ttk.Scrollbar(console, orient="vertical", command=self.console.yview)
        cscroll_x = ttk.Scrollbar(console, orient="horizontal", command=self.console.xview)
        self.console.configure(yscrollcommand=cscroll_y.set, xscrollcommand=cscroll_x.set)
        self.console.grid(row=0, column=0, sticky="nsew")
        cscroll_y.grid(row=0, column=1, sticky="ns")
        cscroll_x.grid(row=1, column=0, sticky="ew")

        settings.columnconfigure(0, weight=1)
        settings.rowconfigure(0, weight=1)
        self.command_text = tk.Text(settings, wrap="word", font=("Consolas", 9))
        self.command_text.grid(row=0, column=0, sticky="nsew")
        self.command_text.insert("1.0", "The exact backend command will appear here before launch.")
        self.command_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Read-only Pipeline Inspector.  Displayed values come from completed
    # backend frames, never from UI guesses or ground-truth feedback.
    # ------------------------------------------------------------------

    def _build_inspector(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=4)
        parent.rowconfigure(5, weight=3)
        parent.rowconfigure(7, weight=3)

        ttk.Label(parent, text="PIPELINE INSPECTOR", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="ew", pady=(0, 2)
        )
        ttk.Label(parent, textvariable=self.var_inspector_frame, font=("Segoe UI", 9, "bold")).grid(
            row=1, column=0, sticky="ew"
        )
        controls = ttk.Frame(parent)
        controls.grid(row=2, column=0, sticky="ew", pady=(4, 5))
        ttk.Checkbutton(
            controls, text="Auto tour", variable=self.var_inspector_auto,
            command=self._inspector_toggle_auto,
        ).pack(side="left")
        ttk.Checkbutton(
            controls, text="Freeze frame", variable=self.var_inspector_freeze,
            command=self._inspector_toggle_freeze,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Export", width=7, command=self._inspector_export).pack(side="right")

        chart_frame = ttk.LabelFrame(parent, text="Flowchart  ·  click a box", padding=3)
        chart_frame.grid(row=3, column=0, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.inspector_chart = tk.Canvas(
            chart_frame, background="#101923", highlightthickness=0, cursor="hand2"
        )
        self.inspector_chart.grid(row=0, column=0, sticky="nsew")
        chart_scroll = ttk.Scrollbar(chart_frame, orient="vertical", command=self.inspector_chart.yview)
        chart_scroll.grid(row=0, column=1, sticky="ns")
        self.inspector_chart.configure(yscrollcommand=chart_scroll.set)
        self.inspector_chart.bind("<Button-1>", self._inspector_chart_click)
        self.inspector_chart.bind("<MouseWheel>", self._inspector_chart_wheel)
        self.inspector_chart.bind("<Button-4>", self._inspector_chart_wheel)
        self.inspector_chart.bind("<Button-5>", self._inspector_chart_wheel)
        self._inspector_draw_flowchart()

        ttk.Label(parent, text="Selected step · purpose / algorithm / source files", font=("Segoe UI", 8, "bold")).grid(
            row=4, column=0, sticky="ew", pady=(5, 1)
        )
        detail_frame = ttk.Frame(parent)
        detail_frame.grid(row=5, column=0, sticky="nsew")
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self.inspector_detail = tk.Text(
            detail_frame, height=9, wrap="word", font=("Segoe UI", 9),
            background="#f0f4f8", foreground="#182331", borderwidth=1,
            relief="solid", padx=6, pady=4,
        )
        self.inspector_detail.grid(row=0, column=0, sticky="nsew")
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.inspector_detail.yview)
        detail_scroll.grid(row=0, column=1, sticky="ns")
        self.inspector_detail.configure(yscrollcommand=detail_scroll.set, state="disabled")

        ttk.Label(parent, text="Real intermediate values for this frame", font=("Segoe UI", 8, "bold")).grid(
            row=6, column=0, sticky="ew", pady=(5, 1)
        )
        values_frame = ttk.Frame(parent)
        values_frame.grid(row=7, column=0, sticky="nsew")
        values_frame.columnconfigure(0, weight=1)
        values_frame.rowconfigure(0, weight=1)
        self.inspector_metrics = ttk.Treeview(
            values_frame, columns=("label", "value"), show="headings", height=7
        )
        self.inspector_metrics.heading("label", text="Measurement")
        self.inspector_metrics.heading("value", text="Latest result")
        self.inspector_metrics.column("label", width=155, minwidth=80, stretch=True)
        self.inspector_metrics.column("value", width=165, minwidth=90, stretch=True)
        self.inspector_metrics.grid(row=0, column=0, sticky="nsew")
        metric_scroll = ttk.Scrollbar(values_frame, orient="vertical", command=self.inspector_metrics.yview)
        metric_scroll.grid(row=0, column=1, sticky="ns")
        self.inspector_metrics.configure(yscrollcommand=metric_scroll.set)
        self.inspector_metrics.bind("<Double-1>", self._inspector_metric_popup)
        ttk.Label(
            parent, textvariable=self.var_inspector_mode, wraplength=325,
            justify="left", font=("Segoe UI", 8),
        ).grid(row=8, column=0, sticky="ew", pady=(3, 0))
        self._inspector_render()

    def _inspector_draw_flowchart(self):
        canvas = self.inspector_chart
        canvas.delete("all")
        self._inspector_stage_items = {}
        box_left, box_right = 17, 309
        box_h, spacing = 38, 12
        for i, step in enumerate(PIPELINE_STAGES):
            y = 8 + i * (box_h + spacing)
            rect = canvas.create_rectangle(
                box_left, y, box_right, y + box_h,
                fill="#273746", outline="#566879", width=1,
                tags=("step", "step_" + step["id"]),
            )
            label = canvas.create_text(
                (box_left + box_right) / 2, y + box_h / 2,
                text="{}. {}  ·  {}".format(i + 1, step["title"], step["short"]),
                font=("Segoe UI", 8, "bold"), fill="#e2ebf2", width=281,
                tags=("step", "step_" + step["id"]),
            )
            self._inspector_stage_items[step["id"]] = (rect, label)
            if i < len(PIPELINE_STAGES) - 1:
                canvas.create_line(
                    (box_left + box_right) / 2, y + box_h,
                    (box_left + box_right) / 2, y + box_h + spacing - 2,
                    fill="#9aafc2", width=1, arrow="last",
                )
        total_h = 8 + len(PIPELINE_STAGES) * (box_h + spacing)
        canvas.configure(scrollregion=(0, 0, 340, total_h))

    def _inspector_chart_click(self, event):
        closest = self.inspector_chart.find_withtag("current")
        if not closest:
            return
        tags = self.inspector_chart.gettags(closest[0])
        for tag in tags:
            if tag.startswith("step_"):
                self._inspector_select_stage(tag[len("step_"):], manual=True)
                return

    def _inspector_chart_wheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -3
        elif getattr(event, "num", None) == 5:
            delta = 3
        else:
            delta = -int(event.delta / 120) * 3 if event.delta else 0
        if delta:
            self.inspector_chart.yview_scroll(delta, "units")
        return "break"

    def _inspector_select_stage(self, stage_id, manual=False):
        if stage_id not in self._inspector_stage_items:
            return
        self.inspector_selected_stage = stage_id
        if manual:
            self.var_inspector_auto.set(False)
        self._inspector_render()

    def _inspector_toggle_auto(self):
        if self.var_inspector_auto.get():
            self.inspector_auto_index = -1
        self._inspector_render()

    def _inspector_toggle_freeze(self):
        if not self.var_inspector_freeze.get() and self.inspector_latest_snapshot:
            self.inspector_display_snapshot = self.inspector_latest_snapshot
        self._inspector_render()

    def _inspector_receive_snapshot(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get("snapshot_kind") != "completed_frame":
            return
        self.inspector_latest_snapshot = snapshot
        if self.var_inspector_freeze.get():
            return
        self.inspector_display_snapshot = snapshot
        if self.var_inspector_auto.get():
            # Tour completed-frame snapshots, one stage per received frame.
            # Do NOT pretend these are execution-time step notifications.
            self.inspector_auto_index = (self.inspector_auto_index + 1) % len(PIPELINE_STAGES)
            self.inspector_selected_stage = PIPELINE_STAGES[self.inspector_auto_index]["id"]
        self._inspector_render()

    def _inspector_render(self):
        if not hasattr(self, "inspector_detail"):
            return
        snapshot = self.inspector_display_snapshot
        selected = self.inspector_selected_stage
        step = next((x for x in PIPELINE_STAGES if x["id"] == selected), PIPELINE_STAGES[0])
        statuses = snapshot.get("stage_status", {})
        for item in PIPELINE_STAGES:
            sid = item["id"]
            rectangle, text_id = self._inspector_stage_items[sid]
            status = statuses.get(sid, "WAITING")
            fill, outline = {
                "COMPLETE": ("#196344", "#54b893"),
                "DISABLED": ("#37404c", "#6a7380"),
                "WARNING": ("#85601e", "#e2bd66"),
                "UNAVAILABLE": ("#4c4e5e", "#8e93a6"),
            }.get(status, ("#273746", "#566879"))
            if sid == selected:
                fill, outline = "#205e9a", "#8bcbff"
            self.inspector_chart.itemconfigure(rectangle, fill=fill, outline=outline, width=2 if sid == selected else 1)
            self.inspector_chart.itemconfigure(text_id, fill="#ffffff")

        frame = snapshot.get("frame")
        carla_frame = snapshot.get("carla_frame")
        if frame is None:
            display_name = "No completed frame yet"
        else:
            display_name = "Frame {} · CARLA {}".format(frame, carla_frame if carla_frame is not None else "?")
        if self.var_inspector_freeze.get():
            display_name += "  [FROZEN]"
        self.var_inspector_frame.set(display_name)
        if self.var_inspector_auto.get():
            self.var_inspector_mode.set("Auto tour: advances one displayed box per completed frame, not live stage timing. Click a box for manual inspection.")
        else:
            self.var_inspector_mode.set("Manual inspection: select any box; values are from one completed synchronized frame.")

        status = statuses.get(selected, "WAITING")
        details = (
            "STEP {} — {}\n".format(PIPELINE_STAGES.index(step) + 1, step["title"])
            + "Status: {}\n\n".format(status)
            + "Purpose: {}\n\n".format(step["purpose"])
            + "Algorithm: {}\n\n".format(step["algorithm"])
            + "Main source module(s):\n  {}\n\n".format("\n  ".join(step["files"]))
            + "Input: {}\n\n".format(step["input"])
            + "Output: {}\n".format(step["output"])
        )
        self.inspector_detail.configure(state="normal")
        self.inspector_detail.delete("1.0", "end")
        self.inspector_detail.insert("1.0", details)
        self.inspector_detail.configure(state="disabled")
        for old in self.inspector_metrics.get_children():
            self.inspector_metrics.delete(old)
        values = snapshot.get("stage_metrics", {}).get(selected, {})
        if isinstance(values, dict) and values:
            for key, value in values.items():
                self.inspector_metrics.insert("", "end", values=(str(key), str(value)))
        else:
            self.inspector_metrics.insert("", "end", values=("Status", "Waiting for backend"))

        # Keep the selected stage visible, but do not unnecessarily scroll on
        # every repaint (so manual scrolling remains usable).
        if selected in self._inspector_stage_items:
            item_id = self._inspector_stage_items[selected][0]
            bounds = self.inspector_chart.bbox(item_id)
            if bounds:
                _, y1, _, y2 = bounds
                view_top = self.inspector_chart.canvasy(0)
                view_h = max(1, self.inspector_chart.winfo_height())
                total_h = max(1, int(self.inspector_chart.cget("scrollregion").split()[-1]))
                if y1 < view_top or y2 > view_top + view_h:
                    target = max(0, y1 - 10)
                    self.inspector_chart.yview_moveto(min(1.0, target / total_h))

    def _inspector_metric_popup(self, _event=None):
        selected = self.inspector_metrics.selection()
        if not selected:
            return
        values = self.inspector_metrics.item(selected[0], "values")
        if len(values) >= 2:
            messagebox.showinfo(str(values[0]), str(values[1]))

    def _inspector_export(self):
        snapshot = self.inspector_display_snapshot
        if not snapshot or snapshot.get("frame") is None:
            messagebox.showinfo("Pipeline Inspector", "No completed frame available to export yet.")
            return
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        folder = output / "live" / "pipeline_inspector_exports"
        folder.mkdir(parents=True, exist_ok=True)
        initial = "pipeline_frame_{:06d}.txt".format(int(snapshot["frame"]))
        location = filedialog.asksaveasfilename(
            title="Export read-only pipeline frame report", initialdir=str(folder),
            initialfile=initial, defaultextension=".txt",
            filetypes=[("Text report", "*.txt"), ("JSON snapshot", "*.json")],
        )
        if not location:
            return
        dest = Path(location)
        try:
            if dest.suffix.lower() == ".json":
                export_data = {
                    "snapshot": snapshot,
                    "stages": list(PIPELINE_STAGES),
                    "selected_stage": self.inspector_selected_stage,
                    "read_only": True,
                }
                dest.write_text(json.dumps(export_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            else:
                lines = [
                    "LVIMOT PIPELINE INSPECTOR — COMPLETED FRAME REPORT",
                    "Frame: {}   CARLA: {}".format(snapshot.get("frame"), snapshot.get("carla_frame")),
                    "Only actual core measurements; unavailable values are not guessed.",
                    "CARLA GT is evaluation-only. Inspector never changes the estimator.",
                    "Sensor branches may execute concurrently; boxes explain data flow.",
                    "=" * 74,
                ]
                for i, step in enumerate(PIPELINE_STAGES, 1):
                    sid = step["id"]
                    lines += [
                        "", "{}. {}  [{}]".format(i, step["title"], snapshot.get("stage_status", {}).get(sid, "WAITING")),
                        "Purpose: " + step["purpose"],
                        "Algorithm: " + step["algorithm"],
                        "Files: " + ", ".join(step["files"]),
                        "Input: " + step["input"],
                        "Output: " + step["output"],
                    ]
                    for key, value in snapshot.get("stage_metrics", {}).get(sid, {}).items():
                        lines.append("  {}: {}".format(key, value))
                dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._append_console("[INSPECTOR] Exported " + str(dest))
            messagebox.showinfo("Pipeline Inspector", "Export saved:\n" + str(dest))
        except Exception as exc:
            messagebox.showerror("Pipeline Inspector", "Unable to export:\n" + str(exc))

    # ------------------------------------------------------------------
    # Reusable control rows
    # ------------------------------------------------------------------

    def _entry_row(self, parent, label, variable, browse=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True)
        self._control_widgets.append(entry)
        if browse is not None:
            ttk.Button(row, text="...", width=3, command=browse).pack(side="left", padx=(4, 0))
        return entry

    def _combo_row(self, parent, label, variable, values, callback=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly")
        combo.pack(side="left", fill="x", expand=True)
        if callback is not None:
            combo.bind("<<ComboboxSelected>>", callback)
        self._control_widgets.append(combo)
        return combo

    def _check_row(self, parent, label, variable):
        check = ttk.Checkbutton(parent, text=label, variable=variable)
        check.pack(anchor="w", pady=2)
        self._control_widgets.append(check)
        return check

    # ------------------------------------------------------------------
    # Presets / validation
    # ------------------------------------------------------------------

    def _preset_changed(self, _event=None):
        self._apply_preset(self.var_preset.get())

    def _apply_preset(self, name):
        preset = PRESETS.get(name)
        if not preset:
            return
        self.var_width.set(str(preset["width"]))
        self.var_height.set(str(preset["height"]))
        self.var_fov.set(str(preset["fov"]))
        self.var_yolo_size.set(str(preset["yolo_size"]))
        self.var_confidence.set(str(preset["confidence"]))
        self.var_vehicles.set(str(preset["vehicles"]))
        self.var_fixed_delta.set(f"{preset['fixed_delta']:.2f}")

    def _numeric_config(self) -> Dict:
        try:
            cfg = {
                "port": int(self.var_port.get()),
                "vehicles": int(self.var_vehicles.get()),
                "seconds": float(self.var_duration.get()),
                "fixed_delta": float(self.var_fixed_delta.get()),
                "seed": int(self.var_seed.get()),
                "sensor_timeout": float(self.var_sensor_timeout.get()),
                "width": int(self.var_width.get()),
                "height": int(self.var_height.get()),
                "fov": float(self.var_fov.get()),
                "yolo_size": int(self.var_yolo_size.get()),
                "confidence": float(self.var_confidence.get()),
                "candidate_hold": int(self.var_candidate_hold.get()),
                "sliding_window_size": int(self.var_sliding_window_size.get()),
                "map_voxel_size": float(self.var_map_voxel_size.get()),
                "map_radius": float(self.var_map_radius.get()),
                "map_max_in_memory": int(self.var_map_max_in_memory.get()),
            }
        except ValueError as exc:
            raise ValueError("One or more numeric settings are invalid") from exc

        if cfg["vehicles"] < 0 or cfg["vehicles"] > 200:
            raise ValueError("NPC vehicles must be between 0 and 200")
        if cfg["seconds"] <= 0:
            raise ValueError("Run seconds must be > 0")
        if cfg["fixed_delta"] <= 0:
            raise ValueError("Fixed delta must be > 0")
        if cfg["sensor_timeout"] < 1:
            raise ValueError("Sensor timeout should be at least 1 second")
        if cfg["width"] < 320 or cfg["height"] < 120:
            raise ValueError("Camera resolution is too small")
        if cfg["width"] > 3840 or cfg["height"] > 2160:
            raise ValueError("Camera resolution is too large for this GUI")
        if not 30.0 <= cfg["fov"] <= 140.0:
            raise ValueError("Camera FOV must be between 30 and 140 degrees")
        if not 0.001 <= cfg["confidence"] <= 1.0:
            raise ValueError("YOLO confidence must be between 0.001 and 1.0")
        if cfg["candidate_hold"] < 0 or cfg["candidate_hold"] > 12:
            raise ValueError("Candidate hold must be between 0 and 12 frames")
        if cfg["sliding_window_size"] < 2 or cfg["sliding_window_size"] > 32:
            raise ValueError("Sliding window must be between 2 and 32")
        if cfg["map_voxel_size"] <= 0.0 or cfg["map_voxel_size"] > 5.0:
            raise ValueError("Voxel size must be > 0 and <= 5 meters")
        if cfg["map_radius"] < 5.0 or cfg["map_radius"] > 500.0:
            raise ValueError("Map radius must be between 5 and 500 meters")
        if cfg["map_max_in_memory"] < 1000:
            raise ValueError("Max in memory must be at least 1000")
        return cfg

    # ------------------------------------------------------------------
    # Calibration / command building
    # ------------------------------------------------------------------

    def _runtime_calibration_path(self) -> Path:
        output_root = Path(self.var_output_root.get())
        if not output_root.is_absolute():
            output_root = self.project_root / output_root
        path = output_root / "gui" / "runtime_calibration.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_runtime_calibration(self, cfg: Dict) -> Path:
        source = Path(os.path.expandvars(os.path.expanduser(self.var_calibration.get())))
        if not source.exists():
            raise FileNotFoundError(f"Calibration file not found: {source}")
        with source.open("r", encoding="utf-8") as f:
            data = json.load(f)

        camera = data.get("camera")
        if not isinstance(camera, dict):
            raise ValueError("Calibration JSON does not contain a camera section")

        width = cfg["width"]
        height = cfg["height"]
        fov = cfg["fov"]
        focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        camera["width"] = int(width)
        camera["height"] = int(height)
        camera["fov"] = float(fov)
        camera["intrinsic"] = [
            [float(focal), 0.0, float(width) / 2.0],
            [0.0, float(focal), float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ]

        data = with_lidar_profile(data, cfg['fixed_delta'])
        path = self._runtime_calibration_path()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def _build_backend_command(self) -> Tuple[List[str], Path]:
        cfg = self._numeric_config()
        if not self.python_exe.exists():
            raise FileNotFoundError(
                f"Live Python environment not found: {self.python_exe}\nRun scripts/setup_live_carla_env.bat first."
            )
        if not self.main_script.exists():
            raise FileNotFoundError(f"LVIMOT command line entry point not found: {self.main_script}")

        runtime_calibration = self._write_runtime_calibration(cfg)
        command = [
            str(self.python_exe),
            "-u",
            str(self.main_script),
            "--live",
            "--carla-install",
            self.var_carla_install.get(),
            "--calibration-file",
            str(runtime_calibration),
            "--host",
            self.var_host.get(),
            "--port",
            str(cfg["port"]),
            "--seconds",
            str(cfg["seconds"]),
            "--wall-seconds",
            str(cfg["seconds"]),
            "--traffic-vehicles",
            str(cfg["vehicles"]),
            "--fixed-delta",
            str(cfg["fixed_delta"]),
            "--seed",
            str(cfg["seed"]),
            "--sensor-timeout",
            str(cfg["sensor_timeout"]),
            "--output-root",
            self.var_output_root.get(),
            "--yolo-model",
            self.var_yolo_model.get(),
            "--yolo-confidence",
            str(cfg["confidence"]),
            "--yolo-image-size",
            str(cfg["yolo_size"]),
            "--yolo-device",
            self.var_yolo_device.get(),
            "--sliding-window-size",
            str(cfg["sliding_window_size"]),
            "--map-voxel-size",
            str(cfg["map_voxel_size"]),
            "--map-radius",
            str(cfg["map_radius"]),
            "--map-max-in-memory",
            str(cfg["map_max_in_memory"]),
        ]
        # GUI mode always enables the backend visualizer stream because the
        # live camera and spatial map are permanent dashboard panels.
        self.var_processed_display.set(True)
        if self.var_autopilot.get():
            command.append("--autopilot")
        if self.var_gt_eval.get():
            command.append("--evaluate-ground-truth")
        return command, runtime_calibration

    def _display_command(self, command: List[str]):
        text = subprocess.list2cmdline(command)
        self.command_text.configure(state="normal")
        self.command_text.delete("1.0", "end")
        self.command_text.insert("1.0", text)
        self.command_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # CARLA server controls
    # ------------------------------------------------------------------

    def _carla_executable(self) -> Path:
        return Path(self.var_carla_install.get()) / ("CarlaUE4.exe" if os.name == "nt" else "CarlaUE4.sh")

    def _is_carla_reachable(self) -> bool:
        try:
            port = int(self.var_port.get())
            with socket.create_connection((self.var_host.get(), port), timeout=0.35):
                return True
        except Exception:
            return False

    def _poll_carla_status(self):
        reachable = self._is_carla_reachable()
        self.var_carla_status.set("CONNECTED" if reachable else "OFFLINE")
        self.root.after(1200, self._poll_carla_status)

    def _check_carla_now(self):
        if self._is_carla_reachable():
            messagebox.showinfo("CARLA", "CARLA server is reachable on the configured host/port.")
        else:
            messagebox.showwarning("CARLA", "CARLA server is not reachable yet.")

    def _start_carla(self):
        if self._is_carla_reachable():
            messagebox.showinfo("CARLA", "CARLA is already running.")
            return
        exe = self._carla_executable()
        if not exe.exists():
            messagebox.showerror("CARLA", f"CARLA executable not found:\n{exe}")
            return
        quality = self.var_render_quality.get()
        args = [str(exe), f"-quality-level={quality}", f"-carla-rpc-port={int(self.var_port.get())}", "-nosound"]
        if not self.var_show_carla_window.get():
            args.extend(['-RenderOffScreen','-ResX=640','-ResY=360'])
        try:
            self.carla_process = subprocess.Popen(args, cwd=str(exe.parent))
            self._append_console("[GUI] Started CARLA: " + subprocess.list2cmdline(args))
        except Exception as exc:
            messagebox.showerror("CARLA", str(exc))

    def _stop_carla(self):
        if self.backend_process is not None and self.backend_process.poll() is None:
            messagebox.showwarning("CARLA", "Stop LVIMOT before stopping CARLA.")
            return
        if self.carla_process is None or self.carla_process.poll() is not None:
            messagebox.showinfo("CARLA", "The GUI did not start a CARLA process that it can stop.")
            return
        try:
            self.carla_process.terminate()
            self._append_console("[GUI] Requested CARLA shutdown.")
        except Exception as exc:
            messagebox.showerror("CARLA", str(exc))

    # ------------------------------------------------------------------
    # Backend controls
    # ------------------------------------------------------------------

    def _start_backend(self):
        if self.backend_process is not None and self.backend_process.poll() is None:
            messagebox.showinfo(APP_TITLE, "LVIMOT is already running.")
            return
        if not self._is_carla_reachable():
            messagebox.showwarning(APP_TITLE, "Start CARLA and wait until it is reachable before starting LVIMOT.")
            return

        try:
            command, runtime_calibration = self._build_backend_command()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._save_config(silent=True)
        self._display_command(command)
        self._archive_previous_live_outputs()
        self._reset_runtime_dashboard()
        self.stop_requested = False
        self.var_backend_status.set("STARTING")
        self.var_last_error.set("")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_controls_enabled(False)

        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Force Unicode output from the Python backend on legacy Windows
        # consoles; tolerate any third-party native bytes in the reader.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env['LVIMOT_DASHBOARD_ONLY'] = '1'
        env["LVIMOT_LIVE_MAP_SNAPSHOT"] = str(self._map_snapshot_path())
        env["LVIMOT_LIVE_CAMERA_SNAPSHOT"] = str(self._camera_snapshot_path())
        env["LVIMOT_CANDIDATE_HOLD_FRAMES"] = str(self._numeric_config()["candidate_hold"])

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self.backend_process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception as exc:
            self.var_backend_status.set("FAILED")
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self._set_controls_enabled(True)
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self.var_backend_status.set("RUNNING")
        self._append_console(f"[GUI] Runtime calibration: {runtime_calibration}")
        self._append_console("[GUI] " + subprocess.list2cmdline(command))

        threading.Thread(target=self._stdout_worker, daemon=True).start()
        threading.Thread(target=self._json_tail_worker, daemon=True).start()

    def _stdout_worker(self):
        process = self.backend_process
        if process is None or process.stdout is None:
            return
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                self.events.put(("stdout", line.rstrip("\r\n")))
        except Exception as exc:
            self.events.put(("error", f"stdout reader: {exc}"))
        finally:
            rc = process.wait()
            self.events.put(("backend_exit", rc))

    def _map_snapshot_path(self) -> Path:
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        path = output / "live" / "live_map_snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _camera_snapshot_path(self) -> Path:
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        path = output / "live" / "live_camera_preview.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _live_jsonl_path(self) -> Path:
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        return output / "live" / "live_frames.jsonl"

    def _json_tail_worker(self):
        path = self._live_jsonl_path()
        offset = 0
        last_size = 0
        while self.backend_process is not None and self.backend_process.poll() is None:
            try:
                if path.exists():
                    size = path.stat().st_size
                    if size < offset or size < last_size:
                        offset = 0
                    last_size = size
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            offset = f.tell()
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            self.events.put(("record", record))
                time.sleep(0.15)
            except Exception as exc:
                self.events.put(("error", f"JSON monitor: {exc}"))
                time.sleep(0.5)

    def _stop_backend(self):
        process = self.backend_process
        if process is None or process.poll() is not None:
            return
        self.stop_requested = True
        self.var_backend_status.set("STOPPING")
        self._append_console("[GUI] Stop requested...")

        def worker():
            try:
                if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                deadline = time.monotonic() + 20.0
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if process.poll() is None:
                    process.terminate()
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Event processing / dashboard updates
    # ------------------------------------------------------------------

    def _process_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "stdout":
                    self._handle_stdout(str(payload))
                elif kind == "record":
                    self._handle_record(payload if isinstance(payload, dict) else {})
                elif kind == "error":
                    self.var_last_error.set(str(payload))
                    self._append_console("[GUI ERROR] " + str(payload))
                elif kind == "backend_exit":
                    self._handle_backend_exit(int(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _handle_stdout(self, line: str):
        self._append_console(line)
        match = FRAME_RE.search(line)
        if not match:
            if "[LVIMOT LIVE ERROR" in line or "Traceback" in line:
                self.var_last_error.set("Backend reported an error. See Console tab.")
            return

        groups = match.groupdict()
        ms = float(groups["ms"])
        self.var_frame.set(groups["frame"])
        self.var_carla_frame.set(groups["carla"])
        self.var_total_ms.set(f"{ms:.0f} ms")
        self.var_fps.set(f"{1000.0 / ms:.2f}" if ms > 0 else "-")
        self.var_rss.set(f"{float(groups['rss']):.0f} MB")
        self.var_tracks.set(groups["tracks"])
        self.var_map.set(groups["map"])
        self.var_stationary.set("YES" if groups["stationary"] == "True" else "NO")
        self.var_sync.set(f"{float(groups['sync']):.2f} ms")
        self.var_camlag.set((groups.get("camlag") or "0") + " f")
        self.var_pump.set(groups.get("pump") or "-")

    def _handle_record(self, record: Dict):
        self.latest_record = record
        self.processed_records += 1
        try:
            self.last_record_frame = int(record.get("frame"))
        except Exception:
            self.last_record_frame = None
        self.var_frame.set(str(record.get("frame", self.var_frame.get())))
        self.var_carla_frame.set(str(record.get("carla_frame", self.var_carla_frame.get())))

        estimate = record.get("state_estimation") or {}
        pose = estimate.get("pose") or []
        velocity = estimate.get("velocity") or []
        if isinstance(pose, list) and len(pose) >= 6:
            roll, pitch, yaw, x, y, z = [self._safe_float(v) for v in pose[:6]]
            self.var_pose.set(f"x={x:8.2f} m   y={y:8.2f} m   z={z:7.2f} m   yaw={math.degrees(yaw):7.2f}°")
            self.pose_history.append((x, y))
            if len(self.pose_history) > 500:
                self.pose_history = self.pose_history[-500:]

        if isinstance(velocity, list) and len(velocity) >= 3:
            vx, vy, vz = [self._safe_float(v) for v in velocity[:3]]
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            self.var_speed.set(f"{speed:.2f} m/s")

        perf = record.get("performance") or {}
        total_ms = perf.get('live_total_ms',perf.get("total_ms"))
        if total_ms is not None:
            total_ms = self._safe_float(total_ms)
            self.var_total_ms.set(f"{total_ms:.0f} ms")
            if total_ms > 0:
                self.var_fps.set(f"{1000.0 / total_ms:.2f}")
        rss = perf.get("rss_mb")
        if rss is not None:
            self.var_rss.set(f"{self._safe_float(rss):.0f} MB")

        tracks = record.get("tracks") or []
        active_tracks = len(tracks)
        self.max_tracks_seen = max(self.max_tracks_seen, active_tracks)
        self.var_tracks.set(str(active_tracks))

        current_frame = self.last_record_frame if self.last_record_frame is not None else self.processed_records
        active_ids = set()
        for track in tracks:
            try:
                tid = int(track.get("track_id"))
            except Exception:
                continue
            active_ids.add(tid)
            self.track_history[tid] = {
                "track": dict(track),
                "last_seen": int(current_frame),
                "active": True,
            }
        for tid, item in list(self.track_history.items()):
            item["active"] = tid in active_ids
            age = int(current_frame) - int(item.get("last_seen", current_frame))
            if age > 120:
                self.track_history.pop(tid, None)

        if active_tracks > 0:
            self.var_tracks_note.set(
                f"{active_tracks} active published track(s) now. The table also retains recent tracks "
                f"for up to 120 processed frames; maximum active at once this run: {self.max_tracks_seen}."
            )
        elif self.track_history:
            self.var_tracks_note.set(
                "No active published tracks in the current frame. Recent tracks are retained below as RECENT "
                "instead of making the table appear empty."
            )
        else:
            self.var_tracks_note.set(
                "No published tracks have been observed yet. Tracks appear only after camera+LiDAR evidence "
                "passes the tracker publication rules."
            )

        mapping = record.get("mapping_4d") or {}
        map_count = mapping.get("total_static_voxels")
        if map_count is not None:
            map_count_int = int(self._safe_float(map_count))
            self.var_map.set(str(map_count_int))
            frame_for_map = self.last_record_frame if self.last_record_frame is not None else self.processed_records
            self.map_history.append((int(frame_for_map), map_count_int))
            if len(self.map_history) > 1200:
                self.map_history = self.map_history[-1200:]
            self.var_map_note.set(
                f"4-D mapper is active with {map_count_int:,} static voxels. "
                "The 4-D Map tab renders a cleaned occupancy-style rolling local BEV; the underlying mapper data is unchanged."
            )

        integrity = record.get("sensor_integrity") or {}
        if "stationary_active" in integrity:
            self.var_stationary.set("YES" if bool(integrity.get("stationary_active")) else "NO")

        sync = record.get("live_sync") or {}
        skew = sync.get("timestamp_skew_s")
        camlag = sync.get("simulation_lag_frames", sync.get("camera_lag_frames"))
        pump = sync.get("world_ticks_last_call")
        if skew is not None:
            self.var_sync.set(f"{1000.0 * self._safe_float(skew):.2f} ms")
        if camlag is not None:
            self.var_camlag.set(f"{int(camlag)} f")
        if pump is not None:
            self.var_pump.set(str(pump))

        camera_ok = sync.get("camera_last_frame") is not None or "camera_dropped" in sync
        lidar_ok = sync.get("lidar_last_frame") is not None or "lidar_dropped" in sync
        imu_ok = sync.get("imu_last_frame") is not None or "imu_dropped" in sync
        dropped = (
            int(sync.get("camera_dropped", 0)),
            int(sync.get("lidar_dropped", 0)),
            int(sync.get("imu_dropped", 0)),
        )
        self.var_sensor.set(
            "RGB={}  LiDAR={}  IMU={}  dropped={}/{}/{}".format(
                "OK" if camera_ok else "?",
                "OK" if lidar_ok else "?",
                "OK" if imu_ok else "?",
                dropped[0], dropped[1], dropped[2],
            )
        )

        self._update_track_table(tracks, pose)
        self._update_health(record)
        self._inspector_receive_snapshot(record.get("pipeline_inspector"))

    def _handle_backend_exit(self, rc: int):
        if self.stop_requested:
            self.var_backend_status.set("STOPPED")
        elif rc == 0:
            self.var_backend_status.set("COMPLETE")
        else:
            self.var_backend_status.set(f"FAILED ({rc})")
            self.var_last_error.set(f"LVIMOT exited with code {rc}. See Console tab.")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_controls_enabled(True)
        self.backend_process = None

    def _update_track_table(self, tracks: List[Dict], pose):
        for item in self.track_table.get_children():
            self.track_table.delete(item)

        ego_x = self._safe_float(pose[3]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        ego_y = self._safe_float(pose[4]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        current_frame = self.last_record_frame if self.last_record_frame is not None else self.processed_records

        rows = []
        for tid, item in self.track_history.items():
            track = item.get("track") or {}
            position = track.get("position") or [0.0, 0.0, 0.0]
            x = self._safe_float(position[0]) if len(position) > 0 else 0.0
            y = self._safe_float(position[1]) if len(position) > 1 else 0.0
            distance = math.hypot(x - ego_x, y - ego_y)
            speed = track.get("speed")
            if speed is None:
                vel = track.get("velocity") or [0.0, 0.0, 0.0]
                speed = math.sqrt(sum(self._safe_float(v) ** 2 for v in vel[:3]))
            last_seen = int(item.get("last_seen", current_frame))
            age = max(0, int(current_frame) - last_seen)
            status = "ACTIVE" if bool(item.get("active")) else "RECENT"
            rows.append((0 if status == "ACTIVE" else 1, age, tid, track, distance, speed, x, y, status))

        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        for _, age, tid, track, distance, speed, x, y, status in rows:
            self.track_table.insert(
                "",
                "end",
                values=(
                    tid,
                    status,
                    track.get("class", "object"),
                    f"{distance:.1f}",
                    f"{self._safe_float(speed):.1f}",
                    f"{x:.1f}",
                    f"{y:.1f}",
                    track.get("motion_state", "-"),
                    track.get("hits", "-"),
                    "now" if age == 0 else f"{age} f ago",
                ),
            )

    # ------------------------------------------------------------------
    # BEV drawing
    # ------------------------------------------------------------------

    def _redraw_bev(self):
        canvas = self.bev_canvas
        if not canvas.winfo_exists():
            return
        width = max(300, canvas.winfo_width())
        height = max(300, canvas.winfo_height())
        canvas.delete("all")

        cx = width / 2.0
        cy = height * 0.72
        px_per_m = min(width / 120.0, height / 100.0)

        for meters in (10, 20, 30, 40, 50):
            radius = meters * px_per_m
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#25384a")
            canvas.create_text(cx + 5, cy - radius + 10, text=f"{meters}m", fill="#6c879d", anchor="w")
        canvas.create_line(0, cy, width, cy, fill="#25384a")
        canvas.create_line(cx, 0, cx, height, fill="#25384a")

        estimate = self.latest_record.get("state_estimation") or {}
        pose = estimate.get("pose") or []
        ego_x = self._safe_float(pose[3]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        ego_y = self._safe_float(pose[4]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        yaw = self._safe_float(pose[2]) if isinstance(pose, list) and len(pose) >= 6 else 0.0

        def world_delta_to_screen(dx: float, dy: float):
            # Heading-up ego-centric display.  Forward is up; left/right is horizontal.
            forward = math.cos(yaw) * dx + math.sin(yaw) * dy
            lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
            return cx + lateral * px_per_m, cy - forward * px_per_m

        # Optional blue trail = recent estimated ego path only.  It is OFF by
        # default because the spatial map is the primary visualization.  Only
        # points inside the visible local radius are drawn, and discontinuous
        # pose jumps break the line instead of stretching across the screen.
        if self.var_show_trajectory.get() and len(self.pose_history) >= 2:
            segment = []
            previous = None
            for x, y in self.pose_history[-120:]:
                dx = x - ego_x
                dy = y - ego_y
                if math.hypot(dx, dy) > 52.0:
                    if len(segment) >= 4:
                        canvas.create_line(*segment, fill="#4ba3ff", width=2)
                    segment = []
                    previous = (x, y)
                    continue
                if previous is not None and math.hypot(x - previous[0], y - previous[1]) > 5.0:
                    if len(segment) >= 4:
                        canvas.create_line(*segment, fill="#4ba3ff", width=2)
                    segment = []
                px, py = world_delta_to_screen(dx, dy)
                segment.extend([px, py])
                previous = (x, y)
            if len(segment) >= 4:
                canvas.create_line(*segment, fill="#4ba3ff", width=2)

        # In heading-up ego coordinates the ego symbol always points upward.
        triangle = [cx, cy - 18, cx - 10, cy + 12, cx + 10, cy + 12]
        canvas.create_polygon(*triangle, fill="#00d084", outline="white")
        canvas.create_text(cx, cy + 25, text="EGO", fill="white")

        tracks = self.latest_record.get("tracks") or []
        for track in tracks:
            pos = track.get("position") or []
            if len(pos) < 2:
                continue
            tx = self._safe_float(pos[0]) - ego_x
            ty = self._safe_float(pos[1]) - ego_y
            px, py = world_delta_to_screen(tx, ty)
            if px < -20 or px > width + 20 or py < -20 or py > height + 20:
                continue
            r = 6
            canvas.create_oval(px - r, py - r, px + r, py + r, fill="#ffb020", outline="white")
            canvas.create_text(px + 9, py - 8, text=f"ID {track.get('track_id', '?')}", fill="white", anchor="w")

        canvas.create_text(
            12, 12,
            text=(
                "Heading-up ego BEV | orange = active tracks"
                + (" | blue = optional recent estimated trail" if self.var_show_trajectory.get() else "")
            ),
            fill="#c8d6e5", anchor="nw", font=("Segoe UI", 10, "bold")
        )
        self.root.after(700, self._redraw_bev)

    def _poll_spatial_map_snapshot(self):
        path = self._map_snapshot_path()
        try:
            if path.exists():
                stat = path.stat()
                mtime_ns = int(stat.st_mtime_ns)
                if self._spatial_snapshot_mtime_ns != mtime_ns:
                    with path.open("r", encoding="utf-8") as handle:
                        snapshot = json.load(handle)
                    if isinstance(snapshot, dict):
                        self.spatial_snapshot = snapshot
                        self._spatial_snapshot_mtime_ns = mtime_ns
        except (OSError, json.JSONDecodeError):
            # The visualizer writes atomically, but tolerate a transient read
            # failure without disturbing the backend.
            pass
        self.root.after(300, self._poll_spatial_map_snapshot)

    def _poll_camera_snapshot(self):
        if not hasattr(self, "camera_canvas") or not self.camera_canvas.winfo_exists():
            return
        path = self._camera_snapshot_path()
        try:
            if path.exists():
                stat = path.stat()
                mtime_ns = int(stat.st_mtime_ns)
                if self._camera_snapshot_mtime_ns != mtime_ns:
                    photo = tk.PhotoImage(file=str(path))
                    canvas_w = max(320, self.camera_canvas.winfo_width() - 20)
                    canvas_h = max(200, self.camera_canvas.winfo_height() - 35)
                    factor = max(
                        1,
                        int(math.ceil(photo.width() / float(canvas_w))),
                        int(math.ceil(photo.height() / float(canvas_h))),
                    )
                    if factor > 1:
                        photo = photo.subsample(factor, factor)
                    self._camera_photo = photo
                    self._camera_snapshot_mtime_ns = mtime_ns
                    self.camera_canvas.delete("all")
                    self.camera_canvas.create_image(
                        max(1, self.camera_canvas.winfo_width()) / 2.0,
                        max(1, self.camera_canvas.winfo_height()) / 2.0,
                        image=self._camera_photo,
                        anchor="center",
                    )
                    self.camera_canvas.create_text(
                        12, 12,
                        text="LIVE EGO RGB | cyan=current fused candidate | dashed cyan=short candidate hold | green=published MOT | amber=track hold",
                        fill="white", anchor="nw", font=("Segoe UI", 10, "bold")
                    )
                    self.var_camera_note.set(
                        "Live CARLA ego RGB is permanently visible. Solid cyan is a current fused camera+LiDAR candidate; "
                        "dashed cyan is a short display-only candidate hold. Green is a real published MOT track; amber is track hold."
                    )
        except (OSError, tk.TclError):
            pass

        if not self.var_processed_display.get() and self._camera_photo is None:
            self.camera_canvas.delete("all")
            self.camera_canvas.create_text(
                max(1, self.camera_canvas.winfo_width()) / 2.0,
                max(1, self.camera_canvas.winfo_height()) / 2.0,
                text="Waiting for the live ego RGB stream...\nStart LVIMOT to populate this permanent camera panel.",
                fill="#8aa0b5", justify="center", width=700,
            )
        self.root.after(350, self._poll_camera_snapshot)

    def _rotate_map_3d(self, event):
        """Mouse drag changes the oblique viewing direction (display only)."""
        if self.var_map_view.get() != "3D height view":
            return
        width = max(100, self.map_canvas.winfo_width())
        angle = (float(event.x) / width - 0.5) * 160.0
        self.var_map_angle.set(max(-80.0, min(80.0, angle)))

    def _map_export_files(self):
        source_dir = self._map_snapshot_path().parent
        return (
            source_dir / "live_map_latest.ply",
            source_dir / "live_map_clean_latest.ply",
            source_dir / "live_dynamic_tracks_latest.json",
        )

    def _save_map_ply(self):
        raw_path, clean_path, tracks_path = self._map_export_files()
        if not raw_path.is_file():
            emergency = raw_path.with_name("live_map_EMERGENCY.ply")
            if emergency.is_file():
                raw_path = emergency
        if not raw_path.is_file():
            messagebox.showwarning(
                APP_TITLE,
                "No full map export exists for this session. Do not rerun CARLA yet. "
                "Check the Console for [MAP EXPORT ERROR]. The old 4-D map "
                "cannot be reconstructed from the sampled GUI preview.",
            )
            return
        selected = filedialog.asksaveasfilename(
            title="Save LVIMOT raw rolling map (and optional clean/tracks files)",
            defaultextension=".ply", filetypes=[("PLY point cloud", "*.ply")],
            initialfile="LVIMOT_rolling_map.ply",
        )
        if not selected:
            return
        target = Path(selected)
        try:
            shutil.copy2(str(raw_path), str(target))
            exports = [target.name]
            if clean_path.is_file():
                clean_target = target.with_name(target.stem + "_clean.ply")
                shutil.copy2(str(clean_path), str(clean_target))
                exports.append(clean_target.name)
            if tracks_path.is_file():
                tracks_target = target.with_name(target.stem + "_dynamic_tracks.json")
                shutil.copy2(str(tracks_path), str(tracks_target))
                exports.append(tracks_target.name)
            # Copy independent CARLA reference comparison when GT evaluation ran.
            for source_name in (
                "CARLA_MAP_VALIDATION.json", "CARLA_vs_LVIMOT_road_overlay.svg",
                "CARLA_reference_road_surface.ply",
                "LVIMOT_aligned_CARLA_frame_EVALUATION_ONLY.ply",
            ):
                reference = raw_path.parent / source_name
                if reference.is_file():
                    dest = target.parent / source_name
                    if dest.resolve() != reference.resolve():
                        shutil.copy2(str(reference), str(dest))
                    exports.append(source_name)
            messagebox.showinfo(
                APP_TITLE,
                "Saved latest available rolling local map:\n" + "\n".join(exports)
                + "\n\nRaw PLY preserves all mapper voxels. Clean PLY is display-filtered."
                + "\nDynamic tracks are saved separately with timestamps.",
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, "Map export failed: {}".format(exc))

    def _redraw_map_health(self):
        if not hasattr(self, "map_canvas") or not self.map_canvas.winfo_exists():
            return
        canvas = self.map_canvas
        width = max(320, canvas.winfo_width())
        height = max(240, canvas.winfo_height())
        canvas.delete("all")
        snapshot = self.spatial_snapshot or {}
        view = self.var_map_view.get()
        range_m = max(15.0, min(60.0, float(self.var_map_zoom.get())))
        ground = snapshot.get("ground_points_local") or []
        structures = snapshot.get("structure_points_local") or []
        measured_road = snapshot.get('measured_road_points_local') or []
        raw_points = snapshot.get("raw_map_points_local") or snapshot.get("map_points_local") or []
        ground_lines = snapshot.get("ground_outlines_local") or []
        structure_lines = snapshot.get("structure_outlines_local") or []
        tracks = snapshot.get("tracks_local") or []
        objects_4d = snapshot.get("individual_objects_local") or []
        cx = float(width) / 2.0
        cy = float(height) * 0.59
        pixels_per_m = min((width - 28.0) / (2.0 * range_m), (height - 68.0) / (1.55 * range_m))

        is_3d = view == "3D height view"
        angle = math.radians(float(self.var_map_angle.get()))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        def project(f, l, z=0.0):
            if is_3d:
                # Oblique 3-D projection: forward and lateral rotate around Z;
                # relative height is visible, without claiming a 3-D mesh.
                u = cos_a * l + sin_a * f
                depth = cos_a * f - sin_a * l
                return cx + u * pixels_per_m, cy - (0.52 * depth + 1.1 * z) * pixels_per_m
            return cx + l * pixels_per_m, cy - f * pixels_per_m

        if not is_3d:
            for radius in (10, 20, 30, 40, 50, 60):
                if radius > range_m:
                    continue
                r = radius * pixels_per_m
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#25384a")
                canvas.create_text(cx + 4, cy - r + 10, text="{}m".format(radius), fill="#6c879d", anchor="w")
            canvas.create_line(0, cy, width, cy, fill="#25384a")
            canvas.create_line(cx, 0, cx, height, fill="#25384a")

        def draw_dots(items, color, size=2, limit=6000, shaded=False):
            if not items:
                return
            stride = max(1, (len(items) + limit - 1) // limit)
            for item in items[::stride]:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                f, l = self._safe_float(item[0]), self._safe_float(item[1])
                if f * f + l * l > range_m * range_m:
                    continue
                z = self._safe_float(item[2]) if len(item) > 2 else 0.0
                x, y = project(f, l, z)
                if -3 <= x < width + 3 and -3 <= y < height + 3:
                    cell_color = color
                    if shaded:
                        cell_color = ("#7193a6" if z < -0.2 else
                                      "#b6cad8" if z < 1.0 else "#e9f0f7")
                    canvas.create_rectangle(x - size, y - size, x + size, y + size,
                                            fill=cell_color, outline=cell_color)

        def draw_outlines(items, color, fill="", line_width=1):
            for line in items:
                if len(line) < 2:
                    continue
                coords = []
                for xy in line:
                    if len(xy) < 2:
                        continue
                    x, y = project(self._safe_float(xy[0]), self._safe_float(xy[1]))
                    coords.extend((x, y))
                if len(coords) < 4:
                    continue
                if len(coords) >= 6:
                    coords.extend(coords[:2])
                    if fill and len(coords) >= 8:
                        canvas.create_polygon(*coords, fill=fill, outline=color, width=line_width)
                        continue
                canvas.create_line(*coords, fill=color, width=line_width)

        if view == "Measured dots":
            draw_dots(raw_points, "#aeb9c4", size=1, limit=6000)
        elif view == "Connected lines":
            draw_outlines(ground_lines, "#324653", line_width=1)
            draw_outlines(structure_lines, "#b9e1ef", line_width=2)
            if not structure_lines and structures:
                draw_dots(structures, "#b9e1ef", size=1)
        elif is_3d:
            # Painter-like ordering keeps taller nearby points readable.
            draw_dots(ground, "#536a79", size=1, limit=3500)
            draw_dots(structures, "#b6cad8", size=2, limit=3500, shaded=True)
            
        else:
            # Filled contiguous areas are easier to read than thousands of dots.
            draw_outlines(ground_lines, "#344f60", fill="#263744")
            draw_outlines(structure_lines, "#a3cbdc", fill="#648396", line_width=1)
            if not ground_lines:
                draw_dots(ground, "#405463", size=2)
            if not structure_lines:
                draw_dots(structures, "#a3cbdc", size=2)

        if measured_road:
            draw_dots(measured_road,'#65c76f',size=1,limit=2500)
        if not is_3d:
            triangle = [cx, cy - 14, cx - 9, cy + 11, cx + 9, cy + 11]
            canvas.create_polygon(*triangle, fill="#00d084", outline="white")
            canvas.create_text(cx, cy + 23, text="EGO", fill="white")
        else:
            x0, y0 = project(0.0, 0.0, 0.0)
            canvas.create_oval(x0 - 6, y0 - 6, x0 + 6, y0 + 6, fill="#00d084", outline="white")
            canvas.create_text(x0, y0 + 19, text="EGO", fill="white")

        # Individual object history is downstream, FGO-derived presentation.
        # Colors: cyan=observed moving, amber=stopped, green=parked-like,
        # gray=last seen. Do not draw a new path during detection gaps.
        represented_ids = set()
        if self.var_show_object_tracks.get():
            for obj in objects_4d:
                state = str(obj.get('motion_status', 'LAST_SEEN'))
                color = ('#49d9f5' if state == 'MOVING' else
                         '#f5b544' if state == 'STOPPED' else
                         '#87e19e' if state == 'PARKED_CANDIDATE' else '#8b929e')
                represented_ids.add(str(obj.get('track_id', '?')))
                path = obj.get('trajectory_local') or []
                segment, previous = [], None
                for xyz in path:
                    if len(xyz) < 2:
                        continue
                    f, l = self._safe_float(xyz[0]), self._safe_float(xyz[1])
                    if f * f + l * l > range_m * range_m:
                        if len(segment) >= 4:
                            canvas.create_line(*segment, fill=color, width=2)
                        segment, previous = [], None
                        continue
                    if previous is not None and math.hypot(f-previous[0], l-previous[1]) > 8.0:
                        if len(segment) >= 4:
                            canvas.create_line(*segment, fill=color, width=2)
                        segment = []
                    x, y = project(f, l, self._safe_float(xyz[2]) if len(xyz) > 2 else 0.0)
                    segment.extend((x, y))
                    previous = (f, l)
                if len(segment) >= 4:
                    canvas.create_line(*segment, fill=color, width=2)

                corners = obj.get('box_corners_local') or []
                if len(corners) == 8:
                    edges = [(0,1),(1,2),(2,3),(3,0)]
                    if is_3d:
                        edges += [(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    for a,b in edges:
                        pa,pb = corners[a],corners[b]
                        if any(len(z)<3 for z in (pa,pb)):
                            continue
                        if max(math.hypot(pa[0],pa[1]), math.hypot(pb[0],pb[1])) > range_m+6:
                            continue
                        x1,y1 = project(*[self._safe_float(v) for v in pa[:3]])
                        x2,y2 = project(*[self._safe_float(v) for v in pb[:3]])
                        canvas.create_line(x1,y1,x2,y2,fill=color,width=2,
                                           dash=(4,3) if state == 'LAST_SEEN' else ())
                center_obj = obj.get('position_local') or []
                if len(center_obj) < 3:
                    continue
                f,l,z = [self._safe_float(v) for v in center_obj[:3]]
                if f*f+l*l > range_m*range_m:
                    continue
                x,y = project(f,l,z)
                speed = self._safe_float(obj.get('speed',0.0))
                canvas.create_text(x+7,y-9,text='ID {} | {} | {:.1f}m/s'.format(
                    obj.get('track_id','?'),state,speed),fill=color,anchor='w',
                    font=('Segoe UI',8,'bold'))

        # Legacy active-track point remains for IDs without an FGO state.
        # It is not treated as a 4D trajectory or metric bounding box.
        for track in tracks:
            if self.var_show_object_tracks.get() and str(track.get('track_id')) in represented_ids:
                continue
            f = self._safe_float(track.get("forward", 0.0))
            l = self._safe_float(track.get("lateral", 0.0))
            if f * f + l * l > range_m * range_m:
                continue
            x, y = project(f, l, 0.0)
            if 0 <= x <= width and 0 <= y <= height:
                predicted = str(track.get("status", "ACTIVE")) != "ACTIVE"
                color = "#f59e0b" if predicted else "#ffb020"
                canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="white")
                canvas.create_text(x + 9, y - 7, text="ID {}{}".format(
                    track.get("track_id", "?"), " P" if predicted else ""), fill="white", anchor="w")

        canvas.create_text(10, 10,
                           text="{} | {:.0f} m".format(view, range_m),
                           fill="#d9e7f1", anchor="nw", font=("Segoe UI", 9, "bold"))
        if not raw_points and not structures and not ground:
            canvas.create_text(width / 2.0, height / 2.0,
                               text="Waiting for the first LVIMOT map snapshot...",
                               fill="#8aa0b5", width=max(280, width - 100), justify="center")
        else:
            stats = snapshot.get("map_render_stats") or {}
            text = "frame {} | raw voxels {:,} | clean cells {:,} | display islands removed {:,}".format(
                snapshot.get("frame", "-"), int(snapshot.get("map_voxels", 0)),
                int(stats.get("clean_cells", len(ground) + len(structures))),
                int(stats.get("removed_display_islands", 0)))
            canvas.create_text(10, height - 8, text=text, fill="#8aa0b5", anchor="sw", font=("Segoe UI", 8))
        measured_score=snapshot.get('road_surface_agreement_pct')
        legacy_score=snapshot.get('legacy_scene_height_agreement_pct')
        road_text=('Road agreement: {:.2f}% | full-scene height check: {:.2f}%'.format(measured_score,legacy_score)
                  if measured_score is not None and legacy_score is not None else
                  'Green: measured road surface')
        canvas.create_text(10,height-27,text=road_text,fill='#85d88d',anchor='sw',font=('Segoe UI',8))
        self.root.after(550, self._redraw_map_health)

    def _update_health(self, record: Dict):
        sync = record.get("live_sync") or {}
        mapping = record.get("mapping_4d") or {}
        estimate = record.get("state_estimation") or {}
        pose = estimate.get("pose") or []
        skew = self._safe_float(sync.get("timestamp_skew_s", 999.0))
        drops = sum(int(sync.get(k, 0) or 0) for k in ("camera_dropped", "lidar_dropped", "imu_dropped"))
        map_count = int(self._safe_float(mapping.get("total_static_voxels", 0)))
        pose_ok = isinstance(pose, list) and len(pose) >= 6
        frame_ok = self.processed_records >= 5
        sync_ok = skew <= 0.002
        map_ok = map_count > 0
        backend_ok = self.var_backend_status.get() in ("RUNNING", "COMPLETE")
        core_ok = backend_ok and frame_ok and sync_ok and map_ok and pose_ok

        if core_ok and drops == 0:
            self.var_health.set("CORE LIVE: PASS")
        elif core_ok:
            self.var_health.set("CORE LIVE: PASS / DROP WARNING")
        else:
            self.var_health.set("CHECKING")

        mot_text = "MOT observed" if self.max_tracks_seen > 0 else "MOT not yet exercised"
        self.var_health_detail.set(
            f"Frames received: {self.processed_records} | exact sensor skew: {1000.0 * skew:.2f} ms | "
            f"sensor drops: {drops} | mapper: {map_count:,} voxels | pose output: {'YES' if pose_ok else 'NO'} | "
            f"{mot_text} (max active tracks this run: {self.max_tracks_seen}). "
            "This proves the live dataflow is operating; quantitative accuracy still requires GT evaluation."
        )

    # ------------------------------------------------------------------
    # Config / file helpers
    # ------------------------------------------------------------------

    def _config_dict(self) -> Dict:
        return {
            "gui_version": 6,
            "carla_install": self.var_carla_install.get(),
            "calibration": self.var_calibration.get(),
            "host": self.var_host.get(),
            "port": self.var_port.get(),
            "render_quality": self.var_render_quality.get(),
            'show_carla_window':self.var_show_carla_window.get(),
            "preset": self.var_preset.get(),
            "autopilot": self.var_autopilot.get(),
            "gt_eval": self.var_gt_eval.get(),
            "processed_display": True,
            "show_trajectory": self.var_show_trajectory.get(),
            "candidate_hold": self.var_candidate_hold.get(),
            "vehicles": self.var_vehicles.get(),
            "duration": self.var_duration.get(),
            "fixed_delta": self.var_fixed_delta.get(),
            "seed": self.var_seed.get(),
            "sensor_timeout": self.var_sensor_timeout.get(),
            "output_root": self.var_output_root.get(),
            "sliding_window_size": self.var_sliding_window_size.get(),
            "map_voxel_size": self.var_map_voxel_size.get(),
            "map_radius": self.var_map_radius.get(),
            "map_max_in_memory": self.var_map_max_in_memory.get(),
            "width": self.var_width.get(),
            "height": self.var_height.get(),
            "fov": self.var_fov.get(),
            "yolo_model": self.var_yolo_model.get(),
            "yolo_size": self.var_yolo_size.get(),
            "confidence": self.var_confidence.get(),
            "yolo_device": self.var_yolo_device.get(),
        }

    def _apply_config_dict(self, data: Dict):
        mapping = {
            "carla_install": self.var_carla_install,
            "calibration": self.var_calibration,
            "host": self.var_host,
            "port": self.var_port,
            "render_quality": self.var_render_quality,
            'show_carla_window':self.var_show_carla_window,
            "preset": self.var_preset,
            "autopilot": self.var_autopilot,
            "gt_eval": self.var_gt_eval,
            "processed_display": self.var_processed_display,
            "show_trajectory": self.var_show_trajectory,
            "candidate_hold": self.var_candidate_hold,
            "vehicles": self.var_vehicles,
            "duration": self.var_duration,
            "fixed_delta": self.var_fixed_delta,
            "seed": self.var_seed,
            "sensor_timeout": self.var_sensor_timeout,
            "output_root": self.var_output_root,
            "sliding_window_size": self.var_sliding_window_size,
            "map_voxel_size": self.var_map_voxel_size,
            "map_radius": self.var_map_radius,
            "map_max_in_memory": self.var_map_max_in_memory,
            "width": self.var_width,
            "height": self.var_height,
            "fov": self.var_fov,
            "yolo_model": self.var_yolo_model,
            "yolo_size": self.var_yolo_size,
            "confidence": self.var_confidence,
            "yolo_device": self.var_yolo_device,
        }
        for key, var in mapping.items():
            if key in data:
                var.set(data[key])
        self.var_processed_display.set(True)

    def _save_config(self, silent=False):
        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self._config_dict(), f, indent=2)
            if not silent:
                messagebox.showinfo(APP_TITLE, f"Saved configuration:\n{self.config_path}")
        except Exception as exc:
            if not silent:
                messagebox.showerror(APP_TITLE, str(exc))

    def _load_saved_config(self, silent=False):
        if not self.config_path.exists():
            if not silent:
                messagebox.showinfo(APP_TITLE, "No saved GUI configuration exists yet.")
            return
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_config_dict(data)
            if int(data.get("gui_version", 0) or 0) < 4:
                # V3 could save the optional blue trail as enabled.  V4 starts
                # old configurations with it off so the spatial map remains the
                # primary visualization unless the user explicitly re-enables it.
                self.var_show_trajectory.set(False)
            if not silent:
                messagebox.showinfo(APP_TITLE, "Configuration loaded.")
        except Exception as exc:
            if not silent:
                messagebox.showerror(APP_TITLE, str(exc))

    def _browse_carla(self):
        path = filedialog.askdirectory(initialdir=self.var_carla_install.get() or str(self.project_root))
        if path:
            self.var_carla_install.set(path)

    def _browse_calibration(self):
        path = filedialog.askopenfilename(
            initialdir=str(Path(self.var_calibration.get()).parent) if self.var_calibration.get() else str(self.project_root),
            filetypes=[("JSON calibration", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.var_calibration.set(path)

    def _archive_previous_live_outputs(self):
        """Prevent a previous run from being replayed into a new GUI session."""
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        live_dir = output / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for name in (
            "live_frames.jsonl", "live_results.json", "live_map_snapshot.json",
            "live_camera_preview.png", "live_map_latest.ply",
            "live_map_clean_latest.ply", "live_dynamic_tracks_latest.json",
            "live_map_EMERGENCY.ply", "CARLA_MAP_VALIDATION.json",
            "CARLA_vs_LVIMOT_road_overlay.svg", "CARLA_reference_road_surface.ply",
            "LVIMOT_aligned_CARLA_frame_EVALUATION_ONLY.ply",
        ):
            path = live_dir / name
            if not path.exists():
                continue
            archived = live_dir / f"{path.stem}_{stamp}{path.suffix}"
            try:
                path.replace(archived)
            except Exception:
                # Never delete unarchived evidence when a file is locked.
                self._append_console("[GUI] Keeping locked previous output: " + str(path))

    def _open_outputs(self):
        path = Path(self.var_output_root.get())
        if not path.is_absolute():
            path = self.project_root / path
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))
        else:
            messagebox.showinfo(APP_TITLE, str(path))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        for widget in self._control_widgets:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="readonly" if enabled else "disabled")
                else:
                    widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _append_console(self, line: str):
        self.console.insert("end", line + "\n")
        # Bound GUI memory during long demonstrations.
        try:
            line_count = int(self.console.index("end-1c").split(".")[0])
            if line_count > 2500:
                self.console.delete("1.0", "500.0")
        except Exception:
            pass
        self.console.see("end")

    def _reset_runtime_dashboard(self):
        self.inspector_latest_snapshot = {}
        self.inspector_display_snapshot = {}
        self.inspector_auto_index = -1
        self.var_inspector_freeze.set(False)
        self._inspector_render()
        self.latest_record = {}
        self.pose_history = []
        self.map_history = []
        self.track_history = {}
        self.spatial_snapshot = {}
        self._spatial_snapshot_mtime_ns = None
        self._camera_snapshot_mtime_ns = None
        self._camera_photo = None
        self.max_tracks_seen = 0
        self.processed_records = 0
        self.last_record_frame = None
        self.var_frame.set("-")
        self.var_carla_frame.set("-")
        self.var_total_ms.set("-")
        self.var_fps.set("-")
        self.var_rss.set("-")
        self.var_tracks.set("0")
        self.var_map.set("0")
        self.var_stationary.set("-")
        self.var_sync.set("-")
        self.var_camlag.set("-")
        self.var_pump.set("-")
        self.var_pose.set("x=-  y=-  z=-  yaw=-")
        self.var_speed.set("-")
        self.var_tracks_note.set("No active published tracks yet.")
        self.var_map_note.set("4-D spatial map waiting for live data.")
        self.var_camera_note.set("Waiting for the first live RGB frame. The camera stays visible across tabs.")
        self.var_health.set("WAITING")
        self.var_health_detail.set("Start LVIMOT to evaluate live system health.")
        for item in self.track_table.get_children():
            self.track_table.delete(item)
        if hasattr(self, "camera_canvas") and self.camera_canvas.winfo_exists():
            self.camera_canvas.delete("all")

    @staticmethod
    def _safe_float(value) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _on_close(self):
        if self.backend_process is not None and self.backend_process.poll() is None:
            if not messagebox.askyesno(APP_TITLE, "LVIMOT is still running. Stop it and close the GUI?"):
                return
            self._stop_backend()
        self.root.after(250, self.root.destroy)


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = tk.Tk()
    app = LVIMOTLiveGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
