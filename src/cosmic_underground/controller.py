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

def _print_inventory(inv_dir: str = "./inventory") -> None:
    import os
    try:
        files = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)")
        return
    print("[INV] Recorded clips:")
    for f in files:
        print("  -", f)


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

        self.mode = "overworld"
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
        
        # Mixer: pass a provider that returns the current session inventory (two defaults + recordings this session)
        self.mixer = Mixer(inventory_provider=lambda: self.model.player.inventory_songs)
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
        self.mode = "overworld"
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
        can_emit = self.audio.broadcast_on and bool(p_tags)
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
                
                # --- Delegate event handling to the active state ---
                self.active_state.handle_event(e)

                # ---- DANCE MODE ----
                if self.mode == "dance":
                    if e.type in (pygame.KEYDOWN, pygame.KEYUP, self.audio.player.boundary_event):
                        self.dance.handle_event(e)
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.dance.handle_event(e)  # let minigame finalize
                        self.mode = "overworld"
                    continue  # swallow events from overworld

                # 3) DAW-mode frame branch
                if self.mode == "mixer":
                    # feed input to DAW first
                    if e.type in (pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL, pygame.MOUSEMOTION):
                        if self.mixer.handle_event(e):
                            continue
                    # swallow ESC in DAW to get back
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE,):
                        try:
                            self.mixer.engine.stop_all()
                            self.mixer.transport.stop()
                        except Exception:
                            pass
                        self.mode = "overworld"
                        try:
                            self.audio.enable_env_playback()
                        except Exception:
                            pass
                        if hasattr(self, "_prev_broadcast_vol") and self._prev_broadcast_vol is not None:
                            try:
                                self.audio.broadcast.set_volume(self._prev_broadcast_vol)
                            except Exception:
                                pass
                            self._prev_broadcast_vol = None
                        continue

                # ---- OVERWORLD ----
                if e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self._win_size = (max(e.w, 640), max(e.h, 360))
                    self.screen = pygame.display.set_mode(self._win_size, pygame.RESIZABLE)
                    C.SCREEN_W, C.SCREEN_H = self.screen.get_size()

                elif e.type == self.audio.player.boundary_event:
                    self.audio.on_boundary_tick()

                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        continue

                # toggle fullscreen on key UP to avoid key-repeat double toggles
                if e.type == pygame.KEYDOWN:
                                            
                    if e.key == pygame.K_n:
                        self.show_panel = not self.show_panel

                    elif e.key == pygame.K_p:
                        if self.show_prompt:
                            self.show_prompt = False
                        else:
                            kind, aid = self.audio.active_source
                            if kind == "zone":
                                z = self.model.map.zones[aid]
                                self.prompt_text = z.loop.prompt if (z.loop and z.loop.prompt) else "(No prompt yet.)"
                            else:
                                p = self.model.map.pois[aid]
                                self.prompt_text = p.loop.prompt if (p.loop and p.loop.prompt) else f"(No prompt yet for {p.name}.)"
                            self.show_prompt = True

                    elif e.key == pygame.K_F5:
                        import importlib, promptgen as _pg  # adjust if promptgen moves into package
                        importlib.reload(_pg)
                        print(f"[PromptGen] Reloaded {getattr(_pg, 'SCHEMA_VERSION', '?')}")
                        k, i = self.audio.active_source
                        if k == "zone":
                            self.model.map.zones[i].loop = None
                            self.audio.request_zone(i, priority=(0, 0, 0), force=True)
                        else:
                            self.model.map.pois[i].loop = None
                            self.audio.request_poi(i, priority=(0, 0, 0), force=True)

                    elif e.key == pygame.K_TAB:
                        # toggle DAW mode
                        if self.mode == "mixer":
                            # leaving mixer: stop DAW audio, resume env, restore broadcast volume
                            try:
                                self.mixer.engine.stop_all()
                                self.mixer.transport.stop()
                            except Exception:
                                pass
                            self.mode = "overworld"
                            try:
                                self.audio.enable_env_playback()
                            except Exception:
                                pass
                            # restore broadcast volume if we muted it on entry
                            if hasattr(self, "_prev_broadcast_vol") and self._prev_broadcast_vol is not None:
                                try:
                                    self.audio.broadcast.set_volume(self._prev_broadcast_vol)
                                except Exception:
                                    pass
                                self._prev_broadcast_vol = None
                        else:
                            # entering mixer: pause/suppress env and mute broadcast so DAW audio is isolated
                            self.mode = "mixer"
                            try:
                                self.audio.disable_env_playback()
                            except Exception:
                                pass
                            try:
                                self._prev_broadcast_vol = getattr(self.audio.broadcast, "volume", None)
                                if self._prev_broadcast_vol is not None:
                                    self.audio.broadcast.set_volume(0.0)
                            except Exception:
                                pass
                        continue
                    
                    elif e.key == pygame.K_k:
                        # Start dance mode on the current audible source if ready
                        k, i = self.audio.active_source
                        loop = None
                        title = ""
                        spec = None
                        if k == "zone":
                            zr = self.model.map.zones[i]
                            loop = zr.loop
                            title = zr.spec.name
                            spec = zr.spec
                        else:
                            p = self.model.map.pois[i]
                            loop = p.loop
                            title = p.name
                            spec = self.model.map.zones[p.zone_id].spec
                        if loop:
                            try:
                                self.audio.player.stop(fade_ms=0)
                            except Exception:
                                pass
                            self.mode = "dance"
                            self.dance.start_for_loop(loop=loop, src_title=title, player=self.audio.player, zone_spec=spec)
                        else:
                            print("[DANCE] No loop ready yet for current source.")
                    
                    elif e.key == pygame.K_t:
                        # cycle env → player → both
                        self.audio.toggle_listen_mode()
                        print("[Mix] mode:", self.audio.listen_mode)
                        continue
                    
                    elif e.key == pygame.K_b:
                        # toggle broadcast on/off (using currently selected index)
                        pl = self.model.player
                        if pl.broadcasting:
                            self.audio.stop_broadcast()
                            pl.broadcasting = False
                        else:
                            if pl.broadcast_index is not None and 0 <= pl.broadcast_index < len(pl.inventory_songs):
                                song = pl.inventory_songs[pl.broadcast_index]
                                self.audio.start_broadcast(song)
                                pl.broadcasting = True
                    
                    elif e.key == pygame.K_0:
                        self.audio.stop_broadcast()
                        self.model.player.broadcasting = False
                    
                    elif e.key in (pygame.K_1, pygame.K_2):
                        slot = 0 if e.key == pygame.K_1 else 1
                        pl = self.model.player
                        if slot < len(pl.inventory_songs):
                            pl.broadcast_index = slot
                            if pl.broadcasting:
                                self.audio.start_broadcast(pl.inventory_songs[slot])  # seamless swap
                    
                    elif e.key == pygame.K_LEFTBRACKET:
                        v = self.audio.broadcast.volume - 0.05
                        self.audio.broadcast.set_volume(v)
                    
                    elif e.key == pygame.K_RIGHTBRACKET:
                        v = self.audio.broadcast.volume + 0.05
                        self.audio.broadcast.set_volume(v)

                    elif e.key == pygame.K_f:
                        # Interact with quest giver if adjacent
                        px, py = self.model.player.tile_x, self.model.player.tile_y
                        q_pid = None
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                tx, ty = px + dx, py + dy
                                pid = self.model.map.pois_at.get((tx, ty))
                                if not pid:
                                    continue
                                poi = self.model.map.pois[pid]
                                if poi.role == "quest_giver":
                                    q_pid = pid
                                    break
                            if q_pid:
                                break

                        if q_pid:
                            if self.model.active_quest is None:
                                npcs = [p for p in self.model.map.pois.values() if p.kind == "npc" and p.name != "Boss Skuggs"]
                                if npcs:
                                    target = random.choice(npcs)
                                    tz = self.model.map.zones[target.zone_id]
                                    self.model.active_quest = Quest(
                                        giver_pid=q_pid,
                                        target_pid=target.pid,
                                        target_name=target.name,
                                        target_tile=target.tile,
                                        target_zone=target.zone_id,
                                        target_zone_name=tz.spec.name,
                                        accepted=True,
                                    )
                            if self.model.active_quest:
                                q = self.model.active_quest
                                tx, ty = q.target_tile
                                self.quest_text = (
                                    f"Find the alien '{q.target_name}'.\n"
                                    f"Location: {q.target_zone_name} at tile ({tx}, {ty}).\n\n"
                                    f"Tip: stand next to them to hear their tune."
                                )
                                self.show_quest = True
                        else:
                            if self.show_quest:
                                self.show_quest = False

                    elif e.key == pygame.K_g:
                        zid = self.model.current_zone_id
                        self.model.map.zones[zid].loop = None
                        self.audio.request_zone(zid, priority=(0, 0, 0), force=True)

                    elif e.key == pygame.K_m:
                        self.cycle_mood()
                        self.audio.request_zone(self.model.current_zone_id, priority=(0, 0, 0), force=True)

                    elif e.key == pygame.K_e:
                        self.edit_mood_text()
                        self.audio.request_zone(self.model.current_zone_id, priority=(0, 0, 0), force=True)

                    elif e.key == pygame.K_r:
                        if self.audio.recorder.is_recording():
                            self.audio.recorder.stop()
                            self.audio.record_armed = False
                            print("[REC] stopped.")
                        else:
                            self.audio.record_armed = True
                            print("[REC] armed: will start at next loop boundary.")

                    elif e.key == pygame.K_i:
                        _print_inventory("./inventory")

            if self.mode == "mixer":
                dt = self.clock.get_time()
                self.mixer.update(dt)
                self.mixer.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(C.FPS)
                continue


            # Movement only in overworld
            if self.mode != "dance":
                keys = pygame.key.get_pressed()
                dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
                dy = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - (keys[pygame.K_UP] or keys[pygame.K_w])
                if dx or dy:
                    self.model.move_player(dx * self.model.player.speed, dy * self.model.player.speed)
                    q = self.model.active_quest
                    if q:
                        px, py = self.model.player.tile_x, self.model.player.tile_y
                        tx, ty = q.target_tile
                        if max(abs(px - tx), abs(py - ty)) <= 1:
                            self.quest_text = f"Quest complete! You found {q.target_name} in {q.target_zone_name}."
                            self.show_quest = True
                            self.model.active_quest = None
                            self.model.quest_completed = True

            # Dance-mode frame branch
            if self.mode == "dance":
                dt = self.clock.get_time()
                self.dance.update(dt)
                self.screen.fill((10, 10, 14))
                self.dance.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(C.FPS)
                continue

            if self.mode != "dance":
                dt_sec = self.clock.get_time() / 1000.0
                # run influence regardless; it decays when disabled
                self.audio.broadcast.apply_influence(world_model=self.model, dt_sec=dt_sec)

            # per-frame broadcast influence
            dt_ms = self.clock.get_time()
            dt_sec = dt_ms / 1000.0
            if getattr(self.audio, "broadcast", None):
                self.audio.broadcast.apply_influence(world_model=self.model, dt_sec=dt_sec)

            # Overworld draw
            self.view.draw(
                self.screen,
                self.audio.record_armed,
                self.audio.recorder.is_recording(),
                self.show_prompt,
                self.prompt_text,
                self.show_quest,
                self.quest_text,               
            )

            # draw top-right music widget
            self.view.draw_music_widget(self.screen, self.audio)

            # --- Delegate update and draw to the active state ---
            dt = self.clock.get_time()
            self.active_state.update(dt)
            self.active_state.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(C.FPS)

        pygame.quit()
