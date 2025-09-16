from __future__ import annotations
import os
import pygame
from typing import Callable, Iterable, Optional
from cosmic_underground.core import config as C
from .state import Project, new_blank_project
from .transport import Transport
from .snapping import grid_label
from .engine import Engine
from .library import scan_inventory

class Mixer:
    def __init__(self, inventory_provider: Optional[Callable[[], Iterable]] = None):
        self.font = pygame.font.SysFont("consolas", 16)
        self.big  = pygame.font.SysFont("consolas", 18, bold=True)

        self.project: Project = new_blank_project()
        self.transport = Transport()
        self.transport.set_from_project(self.project.bpm, self.project.timesig, self.project.loop_region_bars)

        self.grid_div = 4  # 1/4 notes to start
        self.engine = Engine()

        # Inventory (left panel): prefer live session provider (player.inventory_songs),
        # fallback to scanning persistent folder.
        self._inventory_provider = inventory_provider
        self.inventory: list[str] = []
        self._refresh_inventory()

        # viewport
        self.px_per_bar = 120
        self.timeline_scroll = 0.0  # bars

        # drag state
        self.drag_item = None  # Song or raw path string
        self.drag_preview_bar: float | None = None
        self.drag_target_track: int | None = None
        # clip selection/drag state
        self.selected_clip_id: int | None = None
        self._drag_clip = None  # dict with keys: clip, orig_start_bar, orig_track, bar_offset
        # UI buttons / confirmation
        self._btn_clear = pygame.Rect(0, 0, 0, 0)
        self._confirm_clear = False
        self._confirm_yes = pygame.Rect(0, 0, 0, 0)
        self._confirm_no = pygame.Rect(0, 0, 0, 0)

    # ---------- event/update/draw ----------
    def handle_event(self, e) -> bool:
        # derive layout (match draw())
        surf = pygame.display.get_surface()
        W, H = (surf.get_size() if surf else (C.SCREEN_W, C.SCREEN_H))
        top_h = 48
        left_w = 240
        right_w = 220
        track_h = 60
        grid_x0 = left_w
        grid_y0 = top_h
        grid_w  = W - left_w - right_w
        grid_h  = H - top_h - 12

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                # toggle transport; if transitioning to paused, hard-stop any active DAW sounds
                was_playing = self.transport.playing
                self.transport.toggle_play()
                if was_playing and not self.transport.playing:
                    try:
                        self.engine.stop_all()
                    except Exception:
                        pass
                return True
            # loop removed in mixer: no toggle
            if e.key == pygame.K_EQUALS:      # zoom in
                self.px_per_bar = min(480, self.px_per_bar + 20); return True
            if e.key == pygame.K_MINUS:       # zoom out
                self.px_per_bar = max(40, self.px_per_bar - 20); return True
            if e.key == pygame.K_PERIOD:      # nudge forward 1 bar
                self.timeline_scroll += 1.0; return True
            if e.key == pygame.K_COMMA:       # nudge back 1 bar
                self.timeline_scroll = max(0.0, self.timeline_scroll - 1.0); return True
            # delete selected clip
            if e.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if self.selected_clip_id is not None:
                    self._delete_selected_clip(); return True
            # confirm dialog keys
            if self._confirm_clear:
                if e.key in (pygame.K_y, pygame.K_RETURN):
                    self._clear_all_clips(); self._confirm_clear = False; return True
                if e.key in (pygame.K_n, pygame.K_ESCAPE):
                    self._confirm_clear = False; return True

        # start drag from library list
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            mx, my = e.pos
            # confirmation buttons
            if self._confirm_clear:
                if self._confirm_yes.collidepoint(mx, my):
                    self._clear_all_clips(); self._confirm_clear = False; return True
                if self._confirm_no.collidepoint(mx, my):
                    self._confirm_clear = False; return True
            if 0 <= mx < left_w and top_h <= my < H:
                row_y0 = top_h + 32
                if my >= row_y0:
                    idx = int((my - row_y0) // 20)
                    if 0 <= idx < len(self.inventory):
                        self.drag_item = self.inventory[idx]
                        return True
            # clear button
            if self._btn_clear.collidepoint(mx, my):
                self._confirm_clear = True
                return True
            # click on grid: select clip if any and move playhead
            if grid_x0 <= mx < (grid_x0 + grid_w) and grid_y0 <= my < (grid_y0 + grid_h):
                # hit-test clips
                hit = self._clip_at_pos(mx, my, grid_x0, grid_y0, grid_w, track_h)
                b = self.barsnap(self.x_to_bar(mx, grid_x0))
                # notify engine of manual seek; stop any DAW sounds to avoid stale audio
                try:
                    if hasattr(self.engine, 'note_seek'):
                        self.engine.note_seek()
                except Exception:
                    pass
                # stop any currently playing DAW sounds on seek to avoid stale audio, then move playhead
                try:
                    self.engine.stop_all()
                except Exception:
                    pass
                self.transport.seek_to_bar(max(0.0, b))
                if hit is not None:
                    clip, ti = hit
                    self.selected_clip_id = clip.clip_id
                    # start drag with offset so the clip doesn't jump
                    bar_offset = self.x_to_bar(mx, grid_x0) - float(clip.start_bar)
                    self._drag_clip = {
                        "clip": clip,
                        "orig_start_bar": float(clip.start_bar),
                        "orig_track": int(ti),
                        "bar_offset": float(bar_offset),
                    }
                else:
                    self.selected_clip_id = None
                return True

        # update drag preview over grid
        if e.type == pygame.MOUSEMOTION and self.drag_item is not None:
            mx, my = e.pos
            if grid_x0 <= mx < (grid_x0 + grid_w) and grid_y0 <= my < (grid_y0 + grid_h):
                # compute target track
                ti = int((my - grid_y0) // track_h)
                self.drag_target_track = ti if 0 <= ti < len(self.project.tracks) else None
                # snapped bar
                self.drag_preview_bar = self.barsnap(self.x_to_bar(mx, grid_x0))
            else:
                self.drag_target_track = None
                self.drag_preview_bar = None
            return True

        # dragging an existing clip
        if e.type == pygame.MOUSEMOTION and self._drag_clip is not None:
            mx, my = e.pos
            # compute candidate track index and bar
            ti = int((my - grid_y0) // track_h)
            if 0 <= ti < len(self.project.tracks):
                new_bar = self.barsnap(self.x_to_bar(mx, grid_x0) - float(self._drag_clip["bar_offset"]))
                new_bar = max(0.0, new_bar)
                clip = self._drag_clip["clip"]
                clip.start_bar = float(new_bar)
                # if crossing into another track, move the clip between track lists
                prev_ti = int(self._drag_clip["orig_track"])
                if ti != prev_ti:
                    try:
                        if clip in self.project.tracks[prev_ti].clips:
                            self.project.tracks[prev_ti].clips.remove(clip)
                    except Exception:
                        pass
                    if clip not in self.project.tracks[ti].clips:
                        self.project.tracks[ti].clips.append(clip)
                    self._drag_clip["orig_track"] = int(ti)
                clip.track_id = int(ti)
            return True

        # drop to create clip
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag_item is not None:
            try:
                if self.drag_preview_bar is not None and self.drag_target_track is not None:
                    path = getattr(self.drag_item, "wav_path", self.drag_item)
                    # estimate duration (seconds)
                    dur_sec = 4.0
                    try:
                        import soundfile as sf
                        with sf.SoundFile(path) as f:
                            dur_sec = len(f) / float(f.samplerate)
                    except Exception:
                        pass
                    # convert to beats at project BPM
                    beats = max(1.0, dur_sec * (self.project.bpm / 60.0))
                    from .state import Clip, next_clip_id
                    clip = Clip(
                        clip_id=next_clip_id(),
                        track_id=self.drag_target_track,
                        source_path=str(path),
                        start_bar=float(self.drag_preview_bar),
                        length_beats=float(beats),
                    )
                    self.project.tracks[self.drag_target_track].clips.append(clip)
            finally:
                self.drag_item = None
                self.drag_preview_bar = None
                self.drag_target_track = None
            return True

        # end clip drag
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self._drag_clip is not None:
            # already applied live; just clear the drag state
            self._drag_clip = None
            return True

        return False

    def update(self, dt_ms: int):
        self.transport.update()
        self.engine.update(self.project, self.transport)

    def draw(self, screen: pygame.Surface):
        # live-refresh inventory each frame so newly recorded clips appear immediately
        self._refresh_inventory()
        W, H = screen.get_size()
        screen.fill((14, 12, 20))
        top_h = 48
        left_w = 240
        right_w = 220
        track_h = 60
        grid_x0 = left_w
        grid_y0 = top_h
        grid_w  = W - left_w - right_w
        grid_h  = H - top_h - 12

        # --- top bar ---
        pygame.draw.rect(screen, (26,22,36), (0,0,W,top_h))
        title = f"{self.project.title}  |  {self.project.bpm:.0f} BPM  {self.project.timesig[0]}/{self.project.timesig[1]}  |  Grid {grid_label(self.grid_div)}"
        screen.blit(self.big.render(title, True, (230,230,245)), (12, 10))

        btn_w = 28
        play_txt = "■" if self.transport.playing else "▶"
        pygame.draw.rect(screen, (60,50,90), (W//2 - btn_w, 8, btn_w, 32), border_radius=6)
        screen.blit(self.big.render(play_txt, True, (240,240,255)), (W//2 - btn_w + 7, 12))
        # clear button (top bar)
        clr_rect = pygame.Rect(W//2 + 40, 8, 74, 32)
        pygame.draw.rect(screen, (90,50,60), clr_rect, border_radius=6)
        pygame.draw.rect(screen, (200,120,130), clr_rect, width=1, border_radius=6)
        screen.blit(self.font.render("Clear", True, (240,220,220)), (clr_rect.x+14, clr_rect.y+8))
        self._btn_clear = clr_rect

        # --- left library ---
        pygame.draw.rect(screen, (22,20,30), (0, top_h, left_w, H-top_h))
        screen.blit(self.big.render("Inventory", True, (220,220,240)), (12, top_h+8))
        y = top_h + 32
        for path in self.inventory[:12]:
            # Accept either Song-like objects (with wav_path) or raw paths
            p = getattr(path, "wav_path", path)
            name = os.path.basename(str(p))
            screen.blit(self.font.render("• " + name, True, (200,200,220)), (16, y)); y += 20

        # --- right mixer (stub) ---
        pygame.draw.rect(screen, (22,20,30), (W-right_w, top_h, right_w, H-top_h))
        screen.blit(self.big.render("Mixer", True, (220,220,240)), (W-right_w+12, top_h+8))
        y = top_h + 36
        for t in self.project.tracks:
            pygame.draw.rect(screen, (36,32,48), (W-right_w+12, y, right_w-24, 28), border_radius=6)
            screen.blit(self.font.render(t.name, True, t.color), (W-right_w+20, y+6))
            y += 36

        # --- grid bg ---
        pygame.draw.rect(screen, (18,16,26), (grid_x0, grid_y0, grid_w, grid_h))

        # ruler
        bars_visible = int(grid_w / self.px_per_bar) + 2
        first_bar = int(self.timeline_scroll)
        for i in range(bars_visible):
            bar = first_bar + i
            x = grid_x0 + int((bar - self.timeline_scroll) * self.px_per_bar)
            col = (70,70,95) if (bar % 4) else (110,110,150)
            pygame.draw.line(screen, col, (x, grid_y0), (x, grid_y0+grid_h), 1)
            if i % 2 == 0:
                lab = self.font.render(str(bar+1), True, (180,180,210))
                screen.blit(lab, (x+6, grid_y0+4))

        # tracks lanes
        for ti, t in enumerate(self.project.tracks):
            y = grid_y0 + ti*track_h
            pygame.draw.rect(screen, (28,26,40), (grid_x0, y, grid_w, track_h-4))
            # draw clips on this track
            for c in t.clips:
                x = grid_x0 + int((c.start_bar - self.timeline_scroll) * self.px_per_bar)
                bar_len = max(1, self.project.timesig[0])
                w = max(6, int((c.length_beats / bar_len) * self.px_per_bar))
                r = pygame.Rect(x, y+4, w, track_h-12)
                pygame.draw.rect(screen, (60,55,90), r, border_radius=6)
                # selection highlight
                if self.selected_clip_id == getattr(c, 'clip_id', None):
                    pygame.draw.rect(screen, (220,190,120), r, width=2, border_radius=6)
                else:
                    pygame.draw.rect(screen, (140,160,220), r, width=1, border_radius=6)
                name = os.path.basename(getattr(c, 'source_path', 'clip.wav'))
                screen.blit(self.font.render(name, True, (220,220,235)), (r.x+6, r.y+6))
            # track label
            lbl = self.font.render(t.name, True, t.color)
            screen.blit(lbl, (grid_x0+8, y+6))

        # playhead
        ph_x = grid_x0 + int((self.transport.pos_bars - self.timeline_scroll) * self.px_per_bar)
        pygame.draw.line(screen, (255,120,150), (ph_x, grid_y0), (ph_x, grid_y0+grid_h), 2)

        # drag preview indicator
        if self.drag_item is not None and self.drag_preview_bar is not None and self.drag_target_track is not None:
            y = grid_y0 + self.drag_target_track*track_h
            x = grid_x0 + int((self.drag_preview_bar - self.timeline_scroll) * self.px_per_bar)
            w = max(6, int(1.0 * self.px_per_bar))
            r = pygame.Rect(x, y+4, w, track_h-12)
            s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
            s.fill((120,180,255,90))
            screen.blit(s, (r.x, r.y))
            pygame.draw.rect(screen, (200,220,255), r, width=1, border_radius=6)
        # confirmation modal
        if self._confirm_clear:
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((10, 10, 14, 160))
            screen.blit(overlay, (0, 0))
            mw, mh = 360, 140
            mx = (W - mw)//2; my = (H - mh)//2
            modal = pygame.Rect(mx, my, mw, mh)
            pygame.draw.rect(screen, (26,22,36), modal, border_radius=10)
            pygame.draw.rect(screen, (200,160,220), modal, width=2, border_radius=10)
            msg = self.big.render("Clear all clips?", True, (235,235,245))
            screen.blit(msg, (mx + 20, my + 22))
            yes = pygame.Rect(mx + 60, my + 80, 90, 28)
            no  = pygame.Rect(mx + 210, my + 80, 90, 28)
            pygame.draw.rect(screen, (70,55,95), yes, border_radius=6)
            pygame.draw.rect(screen, (70,55,95), no,  border_radius=6)
            pygame.draw.rect(screen, (160,140,220), yes, width=1, border_radius=6)
            pygame.draw.rect(screen, (160,140,220), no,  width=1, border_radius=6)
            screen.blit(self.font.render("Yes (Y)", True, (235,235,245)), (yes.x+10, yes.y+6))
            screen.blit(self.font.render("No (N)",  True, (235,235,245)), (no.x+16,  no.y+6))
            self._confirm_yes = yes
            self._confirm_no = no

    # ---- helpers ----
    def _refresh_inventory(self):
        try:
            if callable(self._inventory_provider):
                items = list(self._inventory_provider())
                # Keep as Song objects or paths; draw() handles either, but store up to a reasonable number
                self.inventory = items
            else:
                self.inventory = scan_inventory()
        except Exception:
            # Never break draw loop on inventory errors
            pass

    def barsnap(self, bars: float) -> float:
        step = 1.0 / max(1, int(self.grid_div))
        return round(float(bars) / step) * step

    def x_to_bar(self, x_px: int, grid_x0: int) -> float:
        return float(self.timeline_scroll) + (float(x_px - grid_x0) / float(self.px_per_bar))

    def _clip_at_pos(self, mx: int, my: int, grid_x0: int, grid_y0: int, grid_w: int, track_h: int):
        # iterate tracks and clips to find topmost hit
        ti = int((my - grid_y0) // track_h)
        if ti < 0 or ti >= len(self.project.tracks):
            return None
        y = grid_y0 + ti*track_h
        # search this track's clips
        clips = self.project.tracks[ti].clips
        for c in reversed(clips):
            x = grid_x0 + int((c.start_bar - self.timeline_scroll) * self.px_per_bar)
            bar_len = max(1, self.project.timesig[0])
            w = max(6, int((c.length_beats / bar_len) * self.px_per_bar))
            r = pygame.Rect(x, y+4, w, track_h-12)
            if r.collidepoint(mx, my):
                return (c, ti)
        return None

    def _delete_selected_clip(self):
        cid = self.selected_clip_id
        if cid is None:
            return
        for t in self.project.tracks:
            for idx, c in enumerate(list(t.clips)):
                if getattr(c, 'clip_id', None) == cid:
                    try:
                        t.clips.pop(idx)
                    except Exception:
                        pass
                    self.selected_clip_id = None
                    try:
                        self.engine.stop_all()
                    except Exception:
                        pass
                    return

    def _clear_all_clips(self):
        for t in self.project.tracks:
            t.clips.clear()
        self.selected_clip_id = None
        try:
            self.engine.stop_all()
        except Exception:
            pass
