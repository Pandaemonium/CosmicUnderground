from typing import List, Optional
import pygame
import numpy as np
from ..core.config import INFO_COLOR, UI_OUTLINE, HILITE, MIN_CLIP_SEC
from ..core.utils import fmt_time, parse_time
from ..audio import mixer as audio_mixer
from ..audio.mixer import stop_all_audio
from ..audio.timeline import TimelineState, ClipEvent, TIMELINE_HEIGHT
# Waveform functionality removed - using Cosmic DAW instead

class StationPanel:
    """Listening Station: inventory list w/ checkboxes, waveform editor + fields,
       audition that stops at selection end, timeline with lane-aware cursor."""
    def __init__(self, fonts, timeline: TimelineState):
        self.FONT, self.SMALL, self.MONO = fonts
        self.timeline = timeline
        self.panel = pygame.Rect(0,0,0,0)
        self.checkbox_boxes: List[pygame.Rect] = []
        self.label_boxes: List[pygame.Rect] = []
        self.selected_idx = 0
        self.pause_btn = pygame.Rect(0,0,0,0)
        self.add_btn = pygame.Rect(0,0,0,0)
        self.play_btn = pygame.Rect(0,0,0,0)
        self.stop_btn = pygame.Rect(0,0,0,0)
        # Waveform functionality removed - using Cosmic DAW instead
        self.wave_rect = pygame.Rect(0, 0, 0, 0)  # Placeholder for layout
        # field editing
        self.start_field = pygame.Rect(0,0,0,0)
        self.end_field = pygame.Rect(0,0,0,0)
        self.active_field: Optional[str] = None
        self.edit_buffer = ""
        # audition tracking
        self.audition_track = None
        self.audition_start_t = 0.0
        self.audition_started_ms = 0
        self.audition_loop = False  # always False: stop at selection end

    def _button(self, rect, label):
        bg = pygame.Surface(rect.size, pygame.SRCALPHA); bg.fill((35,50,90,230))
        return bg, self.SMALL.render(label, True, (220,235,255))

    def render(self, screen, inventory, WIDTH, HEIGHT):
        self.checkbox_boxes.clear(); self.label_boxes.clear()
        # panel size
        w = 540
        list_h = 26 * (len(inventory) + 1) + 60
        panel_h = max(520, min(list_h + 420, HEIGHT - 40))
        self.panel.update(WIDTH - w - 12, 20, w, panel_h)
        s = pygame.Surface(self.panel.size, pygame.SRCALPHA); s.fill((18,26,57,220))
        screen.blit(s, self.panel)
        pygame.draw.rect(screen, UI_OUTLINE, self.panel, 1)
        screen.blit(self.FONT.render("Listening Station", True, (232,240,255)),
                    (self.panel.x + 10, self.panel.y + 8))

        # Pause button
        self.pause_btn.update(self.panel.right - 86, self.panel.y + 6, 76, 24)
        bg, txt = self._button(self.pause_btn, "Pause ⏸")
        screen.blit(bg, self.pause_btn); pygame.draw.rect(screen, HILITE, self.pause_btn, 1)
        screen.blit(txt, (self.pause_btn.x + 10, self.pause_btn.y + 4))
        screen.blit(self.SMALL.render("[Enter]=apply  [C]=Clip  Space=Audition  A=Add@Cursor  ▶/⏹=Timeline",
                                      True, INFO_COLOR), (self.panel.x + 10, self.panel.y + 32))

        # inventory list
        y = self.panel.y + 52
        for i, tr in enumerate(inventory):
            if i == self.selected_idx:
                sel = pygame.Rect(self.panel.x + 6, y - 4, self.panel.w - 12, 24)
                pygame.draw.rect(screen, (30, 40, 70), sel)
            box = pygame.Rect(self.panel.x + 12, y, 18, 18)
            pygame.draw.rect(screen, (210, 220, 235), box, 1)
            if tr.playing:
                pygame.draw.line(screen, (180, 255, 200), (box.x + 3, box.y + 10), (box.x + 8, box.y + 15), 2)
                pygame.draw.line(screen, (180, 255, 200), (box.x + 8, box.y + 15), (box.x + 15, box.y + 5), 2)
            self.checkbox_boxes.append(box)
            label = f"{tr.name}  [{tr.duration:.2f}s]"
            rect = pygame.Rect(box.right + 8, y - 2, self.panel.w - (box.right - self.panel.x) - 20, 22)
            screen.blit(self.SMALL.render(label, True, INFO_COLOR if i != self.selected_idx else (200,225,255)),
                        (rect.x, rect.y))
            self.label_boxes.append(rect)
            y += 24

        # Simplified waveform area (Cosmic DAW handles advanced functionality)
        if inventory:
            tr = inventory[self.selected_idx]
            wave_h = 120
            self.wave_rect.update(self.panel.x + 10, self.panel.y + 52 + 26*max(1, len(inventory)) + 8,
                                  self.panel.w - 20, wave_h)
            
            # Draw simplified waveform area
            pygame.draw.rect(screen, (25, 30, 45), self.wave_rect)
            pygame.draw.rect(screen, UI_OUTLINE, self.wave_rect, 1)
            
            # Draw placeholder text
            placeholder_text = self.SMALL.render("Waveform: Use Cosmic DAW (D key) for advanced editing", True, INFO_COLOR)
            text_rect = placeholder_text.get_rect(center=self.wave_rect.center)
            screen.blit(placeholder_text, text_rect)

            row_y = self.wave_rect.bottom + 8
            screen.blit(self.SMALL.render("Start:", True, INFO_COLOR), (self.wave_rect.x, row_y))
            self.start_field.update(self.wave_rect.x + 52, row_y - 2, 90, 22)
            screen.blit(self.SMALL.render("End:", True, INFO_COLOR),
                        (self.start_field.right + 16, row_y))
            self.end_field.update(self.start_field.right + 56, row_y - 2, 90, 22)

            cur_start = fmt_time(tr.sel_start * tr.duration)
            cur_end = fmt_time(tr.sel_end * tr.duration)

            def draw_field(rect, text, active):
                base = pygame.Surface(rect.size, pygame.SRCALPHA)
                base.fill((22,30,60,220) if not active else (30,44,88,240))
                screen.blit(base, rect)
                pygame.draw.rect(screen, UI_OUTLINE if not active else HILITE, rect, 1)
                shown = text if not active else self.edit_buffer + ("_" if (pygame.time.get_ticks()//500)%2==0 else "")
                screen.blit(self.MONO.render(shown, True, (220,235,255)), (rect.x+6, rect.y+2))

            draw_field(self.start_field, cur_start, self.active_field == "start")
            draw_field(self.end_field,   cur_end,   self.active_field == "end")

        # timeline block
        tl_top = (self.end_field.bottom + 12) if inventory else (self.panel.y + 140)
        self.timeline.rect.update(self.panel.x + 10, tl_top, self.panel.w - 20, TIMELINE_HEIGHT)
        self.timeline.draw(screen, self.SMALL, INFO_COLOR)

        # controls row
        ctrl_y = self.timeline.rect.bottom + 8
        self.add_btn.update(self.timeline.rect.x, ctrl_y, 130, 26)
        bg, txt = self._button(self.add_btn, "Add @ Cursor")
        screen.blit(bg, self.add_btn); pygame.draw.rect(screen, HILITE, self.add_btn, 1)
        screen.blit(txt, (self.add_btn.x + 8, self.add_btn.y + 4))
        self.play_btn.update(self.add_btn.right + 10, ctrl_y, 70, 26)
        self.stop_btn.update(self.play_btn.right + 8, ctrl_y, 70, 26)
        for btn, label in ((self.play_btn, "Play ▶"), (self.stop_btn, "Stop ⏹")):
            bg, txt = self._button(btn, label)
            screen.blit(bg, btn); pygame.draw.rect(screen, HILITE, btn, 1)
            screen.blit(txt, (btn.x + 12, btn.y + 4))

    # ---------- interactions ----------
    def toggle_checkbox_at(self, pos, inventory):
        for i, box in enumerate(self.checkbox_boxes):
            if box.collidepoint(pos):
                tr = inventory[i]
                if tr.playing: tr.stop()
                else: tr.play()
                return True
        return False

    def select_label_at(self, pos):
        for i, lbl in enumerate(self.label_boxes):
            if lbl.collidepoint(pos):
                self.selected_idx = i
                return True
        return False

    def apply_field_edit(self, inventory):
        if not (inventory and self.active_field): return
        tr = inventory[self.selected_idx]
        val = parse_time(self.edit_buffer)
        if val is None: self.active_field = None; return
        val = max(0.0, min(tr.duration, val))
        if self.active_field == "start":
            if (tr.sel_end * tr.duration - val) >= MIN_CLIP_SEC:
                tr.sel_start = val / tr.duration
        else:
            if (val - tr.sel_start * tr.duration) >= MIN_CLIP_SEC:
                tr.sel_end = val / tr.duration
        # Cursor functionality simplified - use Cosmic DAW for advanced features
        self.active_field = None

    def make_clip_from_selection(self, inventory):
        if not inventory: return
        tr = inventory[self.selected_idx]
        t0 = max(0.0, min(tr.sel_start, tr.sel_end)) * tr.duration
        t1 = min(tr.duration, max(tr.sel_start, tr.sel_end) * tr.duration)
        if (t1 - t0) < MIN_CLIP_SEC: return
        s = tr.make_slice_sound(t0, t1)  # temp to get length
        # Convert to numpy array by slicing original (for fade)
        import numpy as np
        from ..core.config import SR
        s0 = int(t0 * SR); s1 = int(t1 * SR)
        arr = tr.array[s0:s1, :].copy()
        fade = min(256, arr.shape[0] // 16)
        if fade > 0:
            w = np.linspace(0, 1, fade)
            for ch in (0,1):
                arr[:fade, ch] = (arr[:fade, ch].astype(np.float32) * w).astype(np.int16)
                arr[-fade:, ch] = (arr[-fade:, ch].astype(np.float32) * w[::-1]).astype(np.int16)
        from ..audio.track import Track
        name = f"Clip of {tr.name} [{t0:.2f}-{t1:.2f}s]"
        inventory.append(Track(name, arr))

    def add_clip_at_cursor(self, inventory, mouse_pos):
        if not inventory: return
        tr = inventory[self.selected_idx]
        t0 = max(0.0, min(tr.sel_start, tr.sel_end)) * tr.duration
        t1 = min(tr.duration, max(tr.sel_start, tr.sel_end) * tr.duration)
        if (t1 - t0) < MIN_CLIP_SEC: return
        lane = self.timeline.active_lane
        self.timeline.events.append(ClipEvent(tr, self.timeline.cursor, t1 - t0, t0, lane))

    # audition: gold playhead, stops at selection end, click to seek
    def audition_start(self, inventory):
        if not inventory: return
        tr = inventory[self.selected_idx]
        start_t = tr.sel_start * tr.duration
        end_t = tr.sel_end * tr.duration
        if start_t >= end_t: return
        snd = tr.make_slice_sound(start_t, end_t)
        audio_mixer.AUDITION_CH.stop()
        audio_mixer.AUDITION_CH.play(snd, loops=0)
        self.audition_track = tr
        self.audition_start_t = start_t
        self.audition_started_ms = pygame.time.get_ticks()
        # Cursor functionality simplified - use Cosmic DAW for advanced features

    def audition_stop(self):
        audio_mixer.AUDITION_CH.stop()
        self.audition_track = None

    def audition_update_cursor(self):
        if not self.audition_track:
            return
        tr = self.audition_track
        end_t = tr.sel_end * tr.duration
        ch = audio_mixer.AUDITION_CH
    
        # If channel stopped externally, finish and snap to end
        if not ch or not ch.get_busy():
            # Cursor functionality simplified - use Cosmic DAW for advanced features
            self.audition_track = None
            return
    
        # advance cursor by elapsed time
        now = pygame.time.get_ticks()
        t = self.audition_start_t + (now - self.audition_started_ms) / 1000.0
        if t >= end_t - 1e-4:
            ch.stop()
            # Cursor functionality simplified - use Cosmic DAW for advanced features
            self.audition_track = None
        else:
            # Cursor functionality simplified - use Cosmic DAW for advanced features
            pass

    # mouse/key handlers
    def handle_mouse_down(self, pos, button, inventory):
        mx, my = pos
        # buttons
        if self.pause_btn.collidepoint(pos):
            stop_all_audio(inventory, self.timeline.active_channels); return
        if self.add_btn.collidepoint(pos):
            self.add_clip_at_cursor(inventory, pos); return
        if self.play_btn.collidepoint(pos):
            self.timeline.start(); return
        if self.stop_btn.collidepoint(pos):
            self.timeline.stop(); return
        # timeline click: set time + active lane or pick clip
        if self.timeline.rect.collidepoint(pos):
            if button == 1:
                # select lane and time
                self.timeline.active_lane = self.timeline.lane_from_y(my)
                self.timeline.cursor = self.timeline.x_to_time(mx)
                return
            elif button == 2:
                self._drag_mode = "pan_tl"; self._drag_x = mx; return
        # waveform interactions simplified - use Cosmic DAW for advanced features
        if inventory and self.wave_rect.collidepoint(pos):
            tr = inventory[self.selected_idx]
            # Simplified interaction - just start audition from selection
            sel_start_t = tr.sel_start * tr.duration
            sel_end_t   = tr.sel_end   * tr.duration
            start_t     = sel_start_t  # Use selection start directly
        
            # always start (or restart) audition from click — single, non-looping slice
            ch = audio_mixer.AUDITION_CH
            if ch:
                snd = tr.make_slice_sound(start_t, sel_end_t)
                ch.stop(); ch.play(snd, loops=0)
            self.audition_track   = tr
            self.audition_start_t = start_t
            self.audition_started_ms = pygame.time.get_ticks()
            return
        # fields focus
        if inventory and self.start_field.collidepoint(pos):
            tr = inventory[self.selected_idx]
            self.active_field = "start"; self.edit_buffer = fmt_time(tr.sel_start * tr.duration); return
        if inventory and self.end_field.collidepoint(pos):
            tr = inventory[self.selected_idx]
            self.active_field = "end"; self.edit_buffer = fmt_time(tr.sel_end * tr.duration); return
        # list
        if self.toggle_checkbox_at(pos, inventory): return
        if self.select_label_at(pos): return

    def handle_mouse_up(self, button):
        self._drag_mode = None

    def handle_mouse_move(self, pos, inventory):
        if getattr(self, "_drag_mode", None) == "pan_tl":
            mx, my = pos
            dx = mx - getattr(self, "_drag_x", mx)
            self._drag_x = mx
            self.timeline.offset = max(0.0, self.timeline.offset - dx / max(1, self.timeline.pps))
            return
        if not inventory: return
        if getattr(self, "_drag_mode", None) in ("sel_start", "sel_end") and self.wave_rect.width > 0:
            mx, my = pos
            trk = inventory[self.selected_idx]
            # Simplified selection adjustment - use Cosmic DAW for advanced features
            u = (mx - self.wave_rect.x) / max(1, self.wave_rect.w)
            u = max(0.0, min(1.0, u))  # Simple 0-1 normalization
            min_norm = MIN_CLIP_SEC / max(1e-6, trk.duration)
            if self._drag_mode == "sel_start":
                trk.sel_start = min(u, trk.sel_end - min_norm)
            else:
                trk.sel_end = max(u, trk.sel_start + min_norm)

    def handle_wheel(self, ev, inventory):
        mx, my = pygame.mouse.get_pos()
        if self.timeline.rect.collidepoint((mx, my)):
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                # pan
                self.timeline.offset = max(0.0, self.timeline.offset - ev.y * (self.timeline.rect.w/self.timeline.pps) * 0.05)
            else:
                # zoom around mouse
                t_center = self.timeline.x_to_time(mx)
                old = self.timeline.pps
                self.timeline.pps = max(20, min(600, int(self.timeline.pps * (1.1 if ev.y > 0 else 0.9))))
                self.timeline.offset = max(0.0, t_center - (mx - self.timeline.rect.x)/max(1, self.timeline.pps))
        elif self.wave_rect.collidepoint((mx, my)) and inventory:
            # Waveform zoom/pan simplified - use Cosmic DAW for advanced features
            pass

    def handle_key(self, ev, inventory, stop_all_fn):
        if self.active_field:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER): self.apply_field_edit(inventory)
            elif ev.key == pygame.K_ESCAPE: self.active_field = None
            elif ev.key == pygame.K_BACKSPACE: self.edit_buffer = self.edit_buffer[:-1]
            else:
                ch = ev.unicode
                if ch and (ch.isdigit() or ch in ".:") and len(self.edit_buffer) < 16:
                    self.edit_buffer += ch
            return
        # no active field
        if ev.key == pygame.K_c: self.make_clip_from_selection(inventory)
        elif ev.key == pygame.K_a: self.add_clip_at_cursor(inventory, pygame.mouse.get_pos())
        elif ev.key == pygame.K_SPACE:
            # toggle audition (stop at selection end)
            if audio_mixer.AUDITION_CH.get_busy(): self.audition_stop()
            else: self.audition_start(inventory)
        elif ev.key == pygame.K_UP:
            self.timeline.active_lane = max(0, self.timeline.active_lane - 1)
        elif ev.key == pygame.K_DOWN:
            self.timeline.active_lane = min(3, self.timeline.active_lane + 1)
