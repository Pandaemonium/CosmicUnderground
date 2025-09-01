from dataclasses import dataclass
from typing import Optional, Dict, List
import pygame
from ..core.config import GRID, CLIP_FILL, CLIP_BORDER, TL_PLAYHEAD
from ..core.utils import fmt_time, nice_grid_step

@dataclass
class ClipEvent:
    track: "Track"
    t_start: float
    t_len: float
    src_start: float
    lane: int
    channel: Optional[pygame.mixer.Channel] = None

TIMELINE_LANES = 4
TIMELINE_HEIGHT = 180

class TimelineState:
    def __init__(self):
        self.rect = pygame.Rect(0,0,0,0)
        self.pps = 120
        self.offset = 0.0
        self.cursor = 0.0
        self.active_lane = 0
        self.is_playing = False
        self.play_started_ms = 0
        self.start_playhead_t = 0.0
        self.events: List[ClipEvent] = []
        self.active_channels: Dict[int, pygame.mixer.Channel] = {}

    # helpers
    def time_to_x(self, t: float) -> int:
        return int(self.rect.x + (t - self.offset) * self.pps)

    def x_to_time(self, x: int) -> float:
        return (x - self.rect.x) / max(1, self.pps) + self.offset

    def lane_y(self, lane: int) -> int:
        lane_h = (TIMELINE_HEIGHT - 20) // TIMELINE_LANES
        return self.rect.y + 20 + lane * lane_h

    def lane_from_y(self, y: int) -> int:
        lane_h = (TIMELINE_HEIGHT - 20) // TIMELINE_LANES
        lane = (y - (self.rect.y + 20)) // lane_h
        return int(max(0, min(TIMELINE_LANES - 1, lane)))

    # transport
    def start(self):
        self.stop()
        self.is_playing = True
        self.start_playhead_t = self.cursor
        self.play_started_ms = pygame.time.get_ticks()

    def stop(self):
        self.is_playing = False
        for ch in list(self.active_channels.values()):
            try: ch.stop()
            except Exception: pass
        self.active_channels.clear()

    # update audio based on time
    def update(self):
        if not self.is_playing: return
        now = pygame.time.get_ticks()
        t = self.start_playhead_t + (now - self.play_started_ms) / 1000.0
        self.cursor = t
        for ev in self.events:
            ev_id = id(ev)
            if ev.t_start <= t < ev.t_start + ev.t_len:
                if ev_id not in self.active_channels:
                    src_t = ev.src_start + (t - ev.t_start)
                    rem = max(0.0, (ev.t_start + ev.t_len) - t)
                    if rem <= 0: continue
                    snd = ev.track.make_slice_sound(src_t, src_t + rem)
                    ch = pygame.mixer.find_channel(True)
                    if ch:
                        ch.play(snd, loops=0)
                        self.active_channels[ev_id] = ch
            else:
                ch = self.active_channels.pop(ev_id, None)
                if ch:
                    try: ch.stop()
                    except Exception: pass

    # drawing
    def draw(self, screen, SMALL, INFO_COLOR):
        import pygame
        from ..core.config import UI_OUTLINE
        pygame.draw.rect(screen, (16,22,45), self.rect)
        pygame.draw.rect(screen, UI_OUTLINE, self.rect, 1)
        # grid & labels
        total_secs = (self.rect.w / max(1, self.pps))
        major = nice_grid_step(self.pps)
        start_tick = (int(self.offset / major)) * major
        t = start_tick
        while t < self.offset + total_secs + major:
            x = self.time_to_x(t)
            pygame.draw.line(screen, GRID, (x, self.rect.y), (x, self.rect.bottom), 1)
            label = fmt_time(t)
            screen.blit(SMALL.render(label, True, INFO_COLOR), (x + 4, self.rect.y + 2))
            t += major
        # lanes
        lane_h = (TIMELINE_HEIGHT - 20) // TIMELINE_LANES
        for i in range(TIMELINE_LANES):
            y = self.lane_y(i)
            color = (90,120,190) if i == self.active_lane else (50,70,110)
            pygame.draw.line(screen, color, (self.rect.x, y), (self.rect.right, y), 1)
            lab = f"Lane {i+1}"
            screen.blit(SMALL.render(lab, True, INFO_COLOR), (self.rect.x + 6, y - lane_h + 4))
        # clips
        for ev in self.events:
            ex = self.time_to_x(ev.t_start)
            ew = max(6, int(ev.t_len * self.pps))
            ey = self.lane_y(ev.lane)
            rect = pygame.Rect(ex, ey - lane_h + 4, ew, lane_h - 8)
            pygame.draw.rect(screen, CLIP_FILL, rect)
            pygame.draw.rect(screen, CLIP_BORDER, rect, 1)
            label = f"{ev.track.name} [{fmt_time(ev.t_len)}]"
            screen.blit(SMALL.render(label, True, (230,240,255)), (rect.x + 6, rect.y + 4))
        # playhead only in active lane (your requested behavior)
        px = self.time_to_x(self.cursor)
        ln_y = self.lane_y(self.active_lane)
        pygame.draw.line(screen, TL_PLAYHEAD, (px, ln_y - lane_h + 4), (px, ln_y - 4), 2)
        screen.blit(SMALL.render(f"t={fmt_time(self.cursor)}", True, (255,220,220)),
                    (px + 6, self.rect.bottom - 18))
