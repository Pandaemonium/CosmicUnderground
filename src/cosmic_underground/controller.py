from __future__ import annotations

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
        self.screen = self._apply_display_mode()
        self.clock = pygame.time.Clock()

        self.model = WorldModel()
        self.audio = AudioService(self.model)
        self.view = GameView(self.model, self.audio)

        self.mode = "overworld"
        self.dance = DanceMinigame(on_finish=self._on_dance_finish)

        self.show_prompt = False
        self.prompt_text = ""
        self.show_panel = False

        self.model.add_tile_changed_listener(lambda _old, _new: setattr(self, "show_prompt", False))
        pygame.key.set_repeat(250, 30)

        self.show_quest = False
        self.quest_text = ""
        if not hasattr(self.model, "active_quest"):
            self.model.active_quest = None

    @property
    def active_quest(self):
        return getattr(self.model, "active_quest", None)

    @active_quest.setter
    def active_quest(self, v):
        self.model.active_quest = v

    def _on_dance_finish(self, res):
        print(f"[DANCE] score={res.score} acc={res.accuracy:.2f} max_combo={res.max_combo} passed={res.passed}")
        self.mode = "overworld"
        q = self.active_quest
        if res.passed and q:
            # Refactor-safe reward: mark quest complete; GameView should render the “complete” sprite.
            self.model.quest_completed = True

            self.quest_text = f"Quest complete! You danced with {q.target_name} in {q.target_zone_name}."
            self.show_quest = True
            self.active_quest = None

    def _adjacent_pois(self) -> List[POI]:
        px, py = self.model.player.tile_x, self.model.player.tile_y
        dt = self.clock.get_time()
        
        # For now, NPCs react to *player* tags only if a player_track exists
        p_tags = self.audio.current_player_tags()
        can_emit = self.audio.broadcast_on and bool(p_tags)
        
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
        if self.fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H), pygame.RESIZABLE)
        C.SCREEN_W, C.SCREEN_H = screen.get_size()
        return screen

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

                # ---- DANCE MODE ----
                if self.mode == "dance":
                    if e.type in (pygame.KEYDOWN, pygame.KEYUP, self.audio.player.boundary_event):
                        self.dance.handle_event(e)
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.dance.handle_event(e)  # let minigame finalize
                        self.mode = "overworld"
                    continue  # swallow events from overworld

                # ---- OVERWORLD ----
                if e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
                    C.SCREEN_W, C.SCREEN_H = self.screen.get_size()

                elif e.type == self.audio.player.boundary_event:
                    self.audio.on_boundary_tick()

                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        continue

                    elif e.key == pygame.K_F11 or (e.key == pygame.K_RETURN and (e.mod & pygame.KMOD_ALT)):
                        self.fullscreen = not self.fullscreen
                        self.screen = self._apply_display_mode()

                    elif e.key == pygame.K_n:
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

            pygame.display.flip()
            self.clock.tick(C.FPS)

        pygame.quit()
