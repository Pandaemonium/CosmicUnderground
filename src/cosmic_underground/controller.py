from __future__ import annotations

from dataclasses import dataclass
import sys
import random
from typing import List
import time
import pygame

from cosmic_underground.core import config as C
from cosmic_underground.core.models import Quest, POI
from cosmic_underground.core.world import WorldModel
from cosmic_underground.audio.service import AudioService
from cosmic_underground.ui.view import GameView
from cosmic_underground.minigames.dance.engine import DanceMinigame
from cosmic_underground.core.affinity import update_npc_affinity, chebyshev
from cosmic_underground.mixer.mixer_ui import Mixer
from cosmic_underground.app.context import GameContext
from cosmic_underground.app.states.game_state import GameState
from cosmic_underground.app.states.overworld_state import OverworldState
from cosmic_underground.app.states.mixer_state import MixerState
from cosmic_underground.app.states.dance_state import DanceState

class GameController:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ – Overworld")
        self.fullscreen = C.DEFAULT_FULLSCREEN
        self._win_size = (C.SCREEN_W, C.SCREEN_H)   # <-- move up here
        self._fs_last_toggle = 0     
        self.screen = self._apply_display_mode()
        self.clock = pygame.time.Clock()

        self.model = WorldModel()
        self.audio = AudioService(self.model)
        self.view = GameView(self.model, self.audio)

        # Game Modes / States
        self.dance = DanceMinigame(on_finish=self._on_dance_finish)
        self.mixer = Mixer(inventory_provider=lambda: self.model.player.inventory_songs)

        # Create the shared context
        self.context = GameContext(
            model=self.model,
            audio=self.audio,
            view=self.view,
            mixer=self.mixer,
            dance_minigame=self.dance,
            controller=self
        )

        self.show_prompt = False
        self.prompt_text = ""
        self.show_panel = False

        self.model.add_tile_changed_listener(lambda _old, _new: setattr(self, "show_prompt", False))
        pygame.key.set_repeat(250, 30)
        # window/FS bookkeeping

        self.show_quest = False
        self.quest_text = ""
        if not hasattr(self.model, "active_quest"):
            self.model.active_quest = None
        
        self.states = {
            "overworld": OverworldState(self.context),
            "mixer": MixerState(self.context),
            "dance": DanceState(self.context),
        }
        self.active_state: GameState = self.states["overworld"]


    @property
    def active_quest(self):
        return getattr(self.model, "active_quest", None)

    @active_quest.setter
    def active_quest(self, v):
        self.model.active_quest = v

    def _on_dance_finish(self, res):
        print(f"[DANCE] score={res.score} acc={res.accuracy:.2f} max_combo={res.max_combo} passed={res.passed}")
        self.change_state("overworld")
        q = self.active_quest
        if res.passed and q:
            # Refactor-safe reward: mark quest complete; GameView should render the “complete” sprite.
            self.model.quest_completed = True

            self.quest_text = f"Quest complete! You danced with {q.target_name} in {q.target_zone_name}."
            self.show_quest = True
            self.active_quest = None

    def change_state(self, new_state_name: str, **kwargs):
        if self.active_state:
            self.active_state.on_exit()
        
        new_state = self.states.get(new_state_name)
        if new_state:
            self.active_state = new_state
            self.active_state.on_enter(**kwargs)

    def _adjacent_pois(self) -> List[POI]:
        px, py = self.model.player.tile_x, self.model.player.tile_y
        dt = self.clock.get_time()
        
        # For now, NPCs react to *player* tags only if a player_track exists
        # NPCs react to player tags if broadcasting is on
        p_tags = self.audio.current_player_tags()
        can_emit = self.audio.broadcast.is_playing() and bool(p_tags)
        
        for poi in self.model.map.pois.values():
            if poi.kind != "npc" or getattr(poi, "mind", None) is None:
                continue
            hear = can_emit and (chebyshev(poi.tile, (px,py)) <= 3)
            update_npc_affinity(poi.mind, dt, hear, p_tags)
        
        out: List[POI] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tx, ty = px + dx, py + dy
                if 0 <= tx < C.MAP_W and 0 <= ty < C.MAP_H:
                    pid = self.model.map.pois_at.get((tx, ty))
                    if pid:
                        out.append(self.model.map.pois[pid])
        return out

    def _apply_display_mode(self):
        def _clamp(sz):
            w, h = sz
            return (max(int(w or 0), 640), max(int(h or 0), 360))

        if self.fullscreen:
            info = pygame.display.Info()
            size = (info.current_w, info.current_h)
            screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
            size = _clamp(self._win_size or (C.SCREEN_W, C.SCREEN_H))
            screen = pygame.display.set_mode(size, pygame.RESIZABLE)

        C.SCREEN_W, C.SCREEN_H = screen.get_size()
        return screen

    
    def _toggle_fullscreen(self):
        now = pygame.time.get_ticks()
        if now - getattr(self, "_fs_last_toggle", 0) < 300:
            return
        self._fs_last_toggle = now

        if not self.fullscreen:
            try:
                self._win_size = self.screen.get_size()
            except Exception:
                pass

        self.fullscreen = not self.fullscreen
        self.screen = self._apply_display_mode()



    def cycle_mood(self):
        order = ["calm", "energetic", "angry", "triumphant", "melancholy", "playful", "brooding", "gritty", "glittery", "funky"]
        zid = self.model.current_zone_id
        z = self.model.map.zones[zid]
        cur = z.spec.mood.lower()
        z.spec.mood = order[(order.index(cur) + 1) % len(order)] if cur in order else "calm"

    def edit_mood_text(self):
        pygame.key.set_repeat(0)
        font = pygame.font.SysFont("consolas", 20)
        entered = ""
        done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN or e.key == pygame.K_ESCAPE:
                        done = True
                    elif e.key == pygame.K_BACKSPACE:
                        entered = entered[:-1]
                    else:
                        if e.unicode and 32 <= ord(e.unicode) < 127:
                            entered += e.unicode
            w, h = self.screen.get_size()
            self.screen.fill((10, 10, 15))
            self.screen.blit(font.render("Enter mood/descriptor (Enter=OK, Esc=Cancel):", True, (240, 240, 240)),
                             (40, h // 2 - 30))
            self.screen.blit(font.render(entered, True, (180, 255, 180)), (40, h // 2 + 10))
            pygame.display.flip()
            self.clock.tick(30)
        if entered.strip():
            self.model.map.zones[self.model.current_zone_id].spec.mood = entered.strip()
        pygame.key.set_repeat(250, 30)

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                # ---- always handle hard-quit ----
                if e.type == pygame.QUIT:
                    running = False
                    continue
                
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.view.handle_music_widget_click(e.pos, self.audio)

                if e.type == pygame.KEYDOWN:
                    # Global fullscreen toggle (works in all modes)
                    if (e.key in (pygame.K_F10, pygame.K_F11)) or (e.key == pygame.K_RETURN and (e.mod & pygame.KMOD_ALT)):
                        self._toggle_fullscreen()
                        continue

                # Handle audio player events that need to be processed at the top level
                if e.type == self.audio.player.fade_finished_event:
                    self.audio.player.handle_fade_finish()
                    continue
                
                # --- Delegate event handling to the active state ---
                self.active_state.handle_event(e)

            # --- Delegate update and draw to the active state ---
            dt = self.clock.get_time()
            self.active_state.update(dt)
            self.active_state.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(C.FPS)

        pygame.quit()
