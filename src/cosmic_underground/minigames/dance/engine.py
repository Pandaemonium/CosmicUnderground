# engine.py
import pygame, math, os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from . import config as C
from .chart import Chart, Note, build_chart_from_metadata, save_chart, load_chart_if_exists, ChartMeta
from .analyzer import build_grid

# --- Key maps (symbolic lane -> pygame key) ---
_PYGAME_KEYS = {
    "A": pygame.K_a, "S": pygame.K_s, "D": pygame.K_d, "F": pygame.K_f,
    "LEFT": pygame.K_LEFT, "DOWN": pygame.K_DOWN, "UP": pygame.K_UP, "RIGHT": pygame.K_RIGHT,
    "SPACE": pygame.K_SPACE,
}
# --- Glyphs for headers & note icons ---
_GLYPH = {
    "A":"A","S":"S","D":"D","F":"F",
    "LEFT":"←","DOWN":"↓","UP":"↑","RIGHT":"→",
    "SPACE":"␣",
}
# --- Colors for per-hit feedback ---
_JCOLOR = {
    "PERFECT": (170,255,170),
    "GREAT":   (170,210,255),
    "GOOD":    (255,215,150),
    "MISS":    (255,130,130),
}

DEFAULT_SPAWN_LEAD_MS = 4000      # visual lead-in
FEEDBACK_MS = 800                 # flyup lifetime
COUNT_IN_BEATS = 8                # 2 bars of 4/4 by default
TAIL_VISIBLE_MS = 200             # how long notes remain after crossing hitline

LEFT_LANE_COLORS = getattr(C, "LEFT_LANE_COLORS", {
    "A": (255, 160, 160),
    "S": (255, 200, 130),
    "D": (160, 220, 255),
    "F": (190, 170, 255),
})


@dataclass
class DanceResult:
    score: int
    accuracy: float
    max_combo: int
    passed: bool

def _space_width_from_config():
    # If user set absolute px width, use it; else multiply base lane width.
    px = getattr(C, "SPACE_LANE_WIDTH", None)
    if isinstance(px, int) and px > 0:
        return px
    mult = getattr(C, "SPACE_LANE_WIDTH_MULT", 1.5)
    try:
        return int(C.LANE_WIDTH * float(mult))
    except Exception:
        return int(C.LANE_WIDTH * 1.5)


# ---------- Single transport / conductor ----------
class Transport:
    def __init__(self, bpm: float, timesig: Tuple[int,int], loop_ms: int,
                 spawn_lead_ms: int, calibration_ms: int, count_in_beats: int = COUNT_IN_BEATS):
        self.bpm = float(bpm)
        self.timesig = timesig
        self.loop_ms = int(loop_ms)
        self.spawn_lead_ms = int(spawn_lead_ms)
        self.calibration_ms = int(calibration_ms)
        self.beat_ms = 60000.0 / max(1e-6, self.bpm)
        self.pre_ms = int(round(count_in_beats * self.beat_ms))
        self.t0_ticks = None
        self.started_audio = False
        # metronome scheduling (during pre-roll only)
        self._next_click_ms = -self.pre_ms

    def start(self):
        # Make song time negative during count-in, cross 0 exactly at music start.
        self.t0_ticks = pygame.time.get_ticks() + self.pre_ms

    def ms_now(self) -> int:
        if self.t0_ticks is None:
            return -self.pre_ms
        return pygame.time.get_ticks() - self.t0_ticks

    def in_precount(self) -> bool:
        return self.ms_now() < 0

    def should_click(self) -> bool:
        """True if we've reached the next pre-roll click time."""
        now = self.ms_now()
        if self._next_click_ms < 0 and now >= self._next_click_ms:
            self._next_click_ms += self.beat_ms
            return True
        return False

# ---------- Dance game ----------
class DanceMinigame:
    def __init__(self, *, on_finish=None, cache_dir: str = "./cache/charts"):
        self.on_finish = on_finish
        self.cache_dir = cache_dir

        # runtime flags/state
        self.active = False
        self.chart: Optional[Chart] = None
        self.loop_ms = 0
        self.transport: Optional[Transport] = None
        self._music_loaded_path: Optional[str] = None

        # scoring
        self.combo = 0
        self.max_combo = 0
        self.score = 0
        self.judgements = {name:0 for (name,_,_) in C.JUDGEMENTS}

        # lane structures
        self._lane_x: Dict[str,int] = {}
        self._hit_y = 0
        self.spawn_y = 90
        self.judge_y = 0
        self.travel_px = 0

        # input buffer (edge-triggered)
        self._pressed = set()

        # fonts
        self.icon_font   = pygame.font.SysFont("consolas", 18, bold=True)
        self.header_font = pygame.font.SysFont("consolas", 20, bold=True)
        self.score_font  = pygame.font.SysFont("consolas", 28, bold=True)
        self.ms_font     = pygame.font.SysFont("consolas", 16, bold=True)

        # feedback flyups
        self.flyups: List[dict] = []

        # click sound (short tick)
        self._click = self._make_click_sound()

        # per-lane indices
        self._lane_times: Dict[str, List[int]] = {}
        self._lane_consumed: Dict[str, List[bool]] = {}
        self._lane_ptr: Dict[str, int] = {}
        self._lane_w: Dict[str, int] = {}

        # options
        self.spawn_lead_ms = getattr(C, "SPAWN_LEAD_MS", DEFAULT_SPAWN_LEAD_MS)
        self.calibration_ms = getattr(C, "CALIBRATION_MS", 0)  # user can set ~ -450 etc.
        self.count_in_beats = getattr(C, "COUNT_IN_BEATS", COUNT_IN_BEATS)

    # --------- API ---------
    def start_for_loop(self, *, loop, src_title: str, player, zone_spec=None, seed: int = 1337):
        """Stops current music, performs a count-in (clicks), then starts the loop at time 0."""
        # Stop whatever the overworld was playing; dance owns the mixer while active.
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        bpm = float(loop.bpm)
        bars = int(getattr(loop, "bars", getattr(zone_spec, "bars", 8)))
        timesig = getattr(loop, "timesig", getattr(zone_spec, "timesig", (4,4)))
        title = src_title or getattr(zone_spec, "name", "Loop")

        # Chart (cache if available)
        provisional = build_chart_from_metadata(
            wav_path=loop.wav_path, title=title, bpm=bpm, bars=bars,
            timesig=timesig, mood=getattr(zone_spec, "mood", "calm"), seed=seed
        )
        cached = load_chart_if_exists(self.cache_dir, provisional.meta)
        self.chart = cached if cached else provisional
        if not cached:
            save_chart(self.cache_dir, self.chart)

        self.loop_ms = int(round(loop.duration_sec * 1000))
        self._music_loaded_path = loop.wav_path

        # Reset scoring and lane structs
        self.combo = self.max_combo = self.score = 0
        for k in self.judgements.keys(): self.judgements[k] = 0
        self._build_lane_indices()

        # Compute layout (needs a real surface)
        self._compute_layout()

        # Transport (single clock)
        self.transport = Transport(
            bpm=bpm,
            timesig=timesig,
            loop_ms=self.loop_ms,
            spawn_lead_ms=self.spawn_lead_ms,
            calibration_ms=self.calibration_ms,
            count_in_beats=self.count_in_beats
        )
        self.transport.start()
        self.active = True
        
    def _lane_width_for(self, lane: str) -> int:
        """SPACE lane is wider; others use base width."""
        if lane == "SPACE":
            return _space_width_from_config()
        return C.LANE_WIDTH
    
    def _lane_bg_color(self, lane: str) -> Tuple[int,int,int]:
        """Colored backgrounds for A/S/D/F; others use neutral BG."""
        return LEFT_LANE_COLORS.get(lane, C.LANE_BG)
    
    def _note_color_for(self, lane: str) -> Tuple[int,int,int]:
        """Match note color to lane color (SPACE keeps strong)."""
        if lane == "SPACE":
            return C.NOTE_STRONG
        return LEFT_LANE_COLORS.get(lane, C.NOTE_COLOR)

    def handle_event(self, e):
        if not self.active:
            return
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self._finish()   # manual exit
                return
            self._pressed.add(e.key)
        elif e.type == pygame.KEYUP:
            self._pressed.discard(e.key)

    def update(self, dt_ms: int):
        if not self.active or not self.chart or not self.transport:
            return

        t = self.transport.ms_now()  # canonical time (can be negative during count-in)

        # Count-in clicks while t < 0
        if self.transport.in_precount():
            if self.transport.should_click():
                self._play_click()
        else:
            # Start music exactly once, right after crossing t >= 0
            if not self.transport.started_audio:
                self._start_music()
                self.transport.started_audio = True

        # Handle inputs (edge-triggered). Judge in transport time + calibration.
        t_judge = t + self.transport.calibration_ms
        for lane_id, key in _PYGAME_KEYS.items():
            if lane_id not in self.chart.lanes:
                continue
            if key in self._pressed:
                self._pressed.discard(key)  # single-shot
                self._judge_lane_press(lane_id, t_judge)

        # Miss sweeper (frame-based)
        self._sweep_misses(t_judge)

        # Expire flyups
        now = pygame.time.get_ticks()
        self.flyups = [f for f in self.flyups if (now - f["created"]) <= FEEDBACK_MS]
    
    def _note_colors(self, lane: str) -> tuple[tuple[int,int,int], tuple[int,int,int]]:
        """
        Return (fill_color, border_color) for note bodies.
        SPACE keeps its strong color; A/S/D/F get bright fill + lane-colored border.
        """
        if lane == "SPACE":
            # strong filled note with light border
            return (C.NOTE_STRONG, (255, 255, 255))
        border = LEFT_LANE_COLORS.get(lane, (220, 220, 255))
        fill   = (245, 245, 255)  # bright/near-white = high contrast on colored lane
        return (fill, border)

    def draw(self, screen: pygame.Surface):
        if not self.active or not self.chart or not self.transport:
            return

        W, H = screen.get_size()
        lanes = self.chart.lanes
        gap = C.LANE_GAP
        
        # widths per lane (SPACE wider)
        widths = [self._lane_width_for(l) for l in lanes]
        self._lane_w = {l: w for l, w in zip(lanes, widths)}
        
        track_w = sum(widths) + gap * (len(lanes) - 1)
        x0 = W//2 - track_w//2
        
        # Background strip
        pygame.draw.rect(screen, C.TRACK_BG, (x0-20, 0, track_w+40, H))
        
        # Lanes + receptor line, with colored left lanes and wide SPACE
        x = x0
        for lane, lw in zip(lanes, widths):
            self._lane_x[lane] = x + lw//2
            # colored BG for A/S/D/F, neutral for others
            bg = self._lane_bg_color(lane)
            pygame.draw.rect(screen, bg, (x, 0, lw, H))
            pygame.draw.rect(screen, C.LANE_BORDER, (x, 0, lw, H), width=2)
            pygame.draw.rect(screen, C.RECEPTOR, (x+10, self._hit_y-4, lw-20, 8), border_radius=4)
            x += lw + gap

        # Lane headers (keys/arrows)
        self._draw_lane_headers(screen)

        # Notes
        t = self.transport.ms_now()
        start_vis = t - TAIL_VISIBLE_MS
        end_vis   = t + self.transport.spawn_lead_ms

        for i, n in enumerate(self.chart.notes):
            if n.time_ms < start_vis or n.time_ms > end_vis:
                continue
            # Skip consumed notes
            lane_list = self._lane_times[n.lane]
            consumed  = self._lane_consumed[n.lane]
            # We need the idx in this lane; precompute a map for speed:
            # (Built once in _build_lane_indices)
            li = self._lane_index_map[i]
            if consumed[li]:
                continue

            y = self._note_screen_y(n.time_ms, t)
            if y is None:
                continue

            lw = self._lane_w.get(n.lane, C.LANE_WIDTH)
            lx = self._lane_x[n.lane] - lw//2 + 12
            rect = pygame.Rect(lx, y - C.NOTE_H//2, lw - 24, C.NOTE_H)
            
            # High-contrast note: bright fill + lane-colored border
            fill, border = self._note_colors(n.lane)
            
            # Optional soft drop shadow for extra separation
            shadow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0, 0, 0, 60), shadow.get_rect(), border_radius=6)
            screen.blit(shadow, (rect.x + 2, rect.y + 2))
            
            # Draw filled body then border
            pygame.draw.rect(screen, fill, rect, border_radius=6)
            pygame.draw.rect(screen, border, rect, width=3, border_radius=6)


            # Round icon with glyph in the center
            cx, cy = rect.center
            r = 11 if n.lane != "SPACE" else 13
            pygame.draw.circle(screen, (16,16,24), (cx, cy), r)
            pygame.draw.circle(screen, (220,220,255), (cx, cy), r, width=2)
            glyph = _GLYPH.get(n.lane, n.lane[:1])
            gsurf = self.icon_font.render(glyph, True, (235,235,255))
            grect = gsurf.get_rect(center=(cx, cy))
            screen.blit(gsurf, grect)

        # Flyups (judgement text + ±ms)
        self._draw_flyups(screen)

        # HUD
        title = f"{self.chart.meta.title}  –  {self.chart.meta.bpm:.0f} BPM"
        screen.blit(self.icon_font.render(title, True, C.TEXT), (20, 16))
        score_s = self.score_font.render(f"Score {self.score}", True, (240,240,255))
        screen.blit(score_s, score_s.get_rect(midtop=(W//2, 10)))
        screen.blit(self.icon_font.render(f"Combo {self.combo}", True, C.TEXT), (20, 40))

        # Accuracy bar
        total_hits = sum(self.judgements.values())
        acc = 0.0 if total_hits == 0 else (
            (self.judgements["PERFECT"]*1.0 + self.judgements["GREAT"]*0.7 + self.judgements["GOOD"]*0.4) / max(1,total_hits)
        )
        ax, ay, aw, ah = 20, 68, 240, 10
        pygame.draw.rect(screen, C.ACC_BAR_BG, (ax, ay, aw, ah), border_radius=5)
        pygame.draw.rect(screen, C.ACC_BAR_OK if acc>=0.7 else C.ACC_BAR_BAD, (ax, ay, int(aw*acc), ah), border_radius=5)

    # --------- internals ---------
    def _compute_layout(self):
        surf = pygame.display.get_surface()
        H = (surf.get_height() if surf else 700)
        self.judge_y = H - C.HITLINE_OFFSET_PX
        self._hit_y  = self.judge_y
        self.spawn_y = 60 + 30
        self.travel_px = max(1, self.judge_y - self.spawn_y)

    def _note_screen_y(self, hit_ms: int, now_ms: int) -> Optional[int]:
        """Map time to vertical position; alpha = 0 at spawn_y, 1 at judge_y."""
        alpha = 1.0 - (hit_ms - now_ms) / float(self.transport.spawn_lead_ms)
        if alpha < 0.0 or alpha > 1.0:
            return None
        return int(self.spawn_y + alpha * self.travel_px)

    def _draw_lane_headers(self, surface: pygame.Surface):
        header_y = 60
        for lane in self.chart.lanes:
            cx = self._lane_x.get(lane)
            if cx is None:
                continue
            text = self.header_font.render(_GLYPH.get(lane, lane), True, (240,240,255))
            pad = 6
            w, h = text.get_width()+pad*2, text.get_height()+pad*2
            rect = pygame.Rect(0, 0, w, h)
            rect.center = (cx, header_y)
            pill = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(pill, (35,35,55,220), pill.get_rect(), border_radius=h//2)
            pygame.draw.rect(pill, (200,200,255,255), pill.get_rect(), width=2, border_radius=h//2)
            pill.blit(text, (pad, pad))
            surface.blit(pill, rect.topleft)

    def _draw_flyups(self, screen: pygame.Surface):
        now = pygame.time.get_ticks()
        for f in self.flyups:
            age = now - f["created"]
            if age > FEEDBACK_MS:
                continue
            t = age / float(FEEDBACK_MS)  # 0..1
            alpha = max(0, 255 - int(255*t))
            # two lines: grade on first, ±ms below (a bit lower to avoid overlap)
            y0 = self._hit_y + 22 + int(18 * t)
            # Grade
            txt = self.icon_font.render(f["grade"], True, f["color"])
            s = pygame.Surface(txt.get_size(), pygame.SRCALPHA)
            s.blit(txt, (0,0)); s.fill((255,255,255,alpha), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(s, s.get_rect(center=(f["x"], y0)).topleft)
            # ±ms
            ms_str = f"{'+' if f['dt_ms']>=0 else ''}{int(f['dt_ms'])} ms"
            ms_txt = self.ms_font.render(ms_str, True, (220,220,255))
            s2 = pygame.Surface(ms_txt.get_size(), pygame.SRCALPHA)
            s2.blit(ms_txt, (0,0)); s2.fill((255,255,255,alpha), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(s2, s2.get_rect(center=(f["x"], y0 + 16)).topleft)

    def _build_lane_indices(self):
        """Pre-split notes by lane for fast nearest search & sweeping."""
        assert self.chart is not None
        self._lane_times = {lane: [] for lane in self.chart.lanes}
        self._lane_consumed = {lane: [] for lane in self.chart.lanes}
        # Map global note index -> lane-local index
        self._lane_index_map: Dict[int, int] = {}

        lane_counts = {lane: 0 for lane in self.chart.lanes}
        for gi, n in enumerate(self.chart.notes):
            lst = self._lane_times[n.lane]
            lst.append(n.time_ms)
            self._lane_consumed[n.lane].append(False)
            self._lane_index_map[gi] = lane_counts[n.lane]
            lane_counts[n.lane] += 1

        self._lane_ptr = {lane: 0 for lane in self.chart.lanes}

    def _judge_lane_press(self, lane: str, t_judge: float):
        """Pick nearest unconsumed note on this lane and grade against C.JUDGEMENTS."""
        times = self._lane_times[lane]
        used  = self._lane_consumed[lane]
        ptr   = self._lane_ptr[lane]
        if not times:
            # stray press on lane with no notes
            self._penalize_stray()
            return

        # nearest search around ptr
        max_window = max(w for _,w,_ in C.JUDGEMENTS)
        best_i, best_dt = None, None

        # scan forward while within window or until dt starts increasing past window
        i = ptr
        while i < len(times):
            if used[i]:
                i += 1; continue
            dt = t_judge - times[i]
            adt = abs(dt)
            if best_dt is None or adt < abs(best_dt):
                best_dt, best_i = dt, i
            # small optimization: if times[i] well beyond t_judge+max_window, we can stop
            if times[i] > t_judge + max_window:
                break
            i += 1

        # also peek one step behind ptr (in case nearest is slightly earlier)
        j = max(0, ptr-1)
        while j < ptr:
            if not used[j]:
                dt = t_judge - times[j]
                if best_dt is None or abs(dt) < abs(best_dt):
                    best_dt, best_i = dt, j
            j += 1

        if best_i is None:
            self._penalize_stray()
            return

        name, window, points = next((n,w,p) for (n,w,p) in C.JUDGEMENTS if abs(best_dt) <= w)
        x = self._lane_x.get(lane, 0)
        self._spawn_flyup(x, name, best_dt)

        if name == "MISS":
            # Don’t consume if outside MISS window; treat as stray.
            self._penalize_stray()
            return

        # GOOD or better → consume and score
        used[best_i] = True
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        self.score += points

        # advance ptr while consumed
        while self._lane_ptr[lane] < len(times) and used[self._lane_ptr[lane]]:
            self._lane_ptr[lane] += 1

    def _sweep_misses(self, t_judge: float):
        """Mark notes that have passed beyond MISS window without being hit."""
        miss_window = max(w for _,w,_ in C.JUDGEMENTS)  # the 'MISS' entry should be last, but we use max()
        for lane in self.chart.lanes:
            times = self._lane_times[lane]
            used  = self._lane_consumed[lane]
            ptr   = self._lane_ptr[lane]
            while ptr < len(times):
                if used[ptr]:
                    ptr += 1; continue
                if t_judge > (times[ptr] + miss_window):
                    # Miss it
                    used[ptr] = True
                    self.combo = 0
                    x = self._lane_x.get(lane, 0)
                    self._spawn_flyup(x, "MISS", t_judge - times[ptr])
                    ptr += 1
                else:
                    break
            self._lane_ptr[lane] = ptr

    def _spawn_flyup(self, x: int, grade: str, dt_ms: float):
        self.flyups.append({
            "created": pygame.time.get_ticks(),
            "grade": grade,
            "dt_ms": dt_ms,
            "x": x,
            "color": _JCOLOR.get(grade, (230,230,230)),
        })

    def _penalize_stray(self):
        self.combo = 0
        # small health system could go here if you add one later

    def _make_click_sound(self, sr: int = 44100, dur_ms: int = 55) -> pygame.mixer.Sound:
        """Small tick with fast decay."""
        n = int(sr * (dur_ms / 1000.0))
        t = np.linspace(0, dur_ms/1000.0, n, endpoint=False)
        freq = 1200.0
        wave = (np.sin(2*np.pi*freq*t) * np.exp(-t*30.0)).astype(np.float32)
        # stereo 16-bit
        s16 = np.clip(wave * 3000, -32768, 32767).astype(np.int16)
        stereo = np.stack([s16, s16], axis=1).tobytes()
        return pygame.mixer.Sound(buffer=stereo)

    def _play_click(self):
        try:
            self._click.play()
        except Exception:
            pass

    def _start_music(self):
        # Start loop, aligned to transport t=0 (audio output latency is handled via calibration_ms)
        try:
            if self._music_loaded_path and os.path.isfile(self._music_loaded_path):
                pygame.mixer.music.load(self._music_loaded_path)
                pygame.mixer.music.play(loops=-1, fade_ms=0)
        except Exception as e:
            print(f"[DANCE] could not start music: {e}")

    def _finish(self):
        self.active = False
        res = self._result()
        if callable(self.on_finish):
            self.on_finish(res)

    def _result(self) -> DanceResult:
        total = sum(self.judgements.values())
        acc = 0.0 if total == 0 else (
            (self.judgements["PERFECT"]*1.0 + self.judgements["GREAT"]*0.7 + self.judgements["GOOD"]*0.4) / max(1,total)
        )
        passed = (acc >= 0.65) or (self.score >= 2500)
        return DanceResult(score=self.score, accuracy=acc, max_combo=self.max_combo, passed=passed)
