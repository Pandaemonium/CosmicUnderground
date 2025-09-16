from __future__ import annotations
import pygame
import random

from cosmic_underground.core import config as C
from cosmic_underground.core.models import Quest
from .game_state import GameState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmic_underground.app.context import GameContext

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

class OverworldState(GameState):
    def __init__(self, context: GameContext):
        super().__init__(context)
        self._key_handlers = {
            pygame.K_ESCAPE: self._quit_game,
            pygame.K_q: self._quit_game,
            pygame.K_n: self._toggle_panel,
            pygame.K_p: self._toggle_prompt,
            pygame.K_TAB: self._enter_mixer,
            pygame.K_k: self._enter_dance_mode,
            pygame.K_i: lambda: _print_inventory("./inventory"),
            pygame.K_f: self._handle_interaction,
            pygame.K_g: self._regenerate_zone_audio,
            pygame.K_m: self._cycle_mood,
            pygame.K_e: self._edit_mood,
            pygame.K_r: self._toggle_record,
            pygame.K_t: self._toggle_listen_mode,
            pygame.K_b: self._toggle_broadcast,
            pygame.K_0: self._stop_broadcast,
            pygame.K_1: lambda: self._select_broadcast_slot(0),
            pygame.K_2: lambda: self._select_broadcast_slot(1),
            pygame.K_LEFTBRACKET: lambda: self._adjust_broadcast_volume(-0.05),
            pygame.K_RIGHTBRACKET: lambda: self._adjust_broadcast_volume(0.05),
            pygame.K_F5: self._reload_promptgen_and_regen,
        }

    def handle_event(self, e: pygame.event.Event):
        ctrl = self.context.controller
        if e.type == pygame.VIDEORESIZE and not ctrl.fullscreen:
            ctrl._win_size = (max(e.w, 640), max(e.h, 360))
            ctrl.screen = pygame.display.set_mode(ctrl._win_size, pygame.RESIZABLE)
            C.SCREEN_W, C.SCREEN_H = ctrl.screen.get_size()
        elif e.type == self.context.audio.player.boundary_event:
            self.context.audio.on_boundary_tick()
        elif e.type == pygame.KEYDOWN:
            handler = self._key_handlers.get(e.key)
            if handler:
                handler()

    def _quit_game(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _toggle_panel(self):
        self.context.controller.show_panel = not self.context.controller.show_panel

    def _toggle_prompt(self):
        ctrl = self.context.controller
        if ctrl.show_prompt:
            ctrl.show_prompt = False
        else:
            audio = self.context.audio
            model = self.context.model
            kind, aid = audio.active_source
            if kind == "zone":
                z = model.map.zones[aid]
                ctrl.prompt_text = z.loop.prompt if (z.loop and z.loop.prompt) else "(No prompt yet.)"
            else:
                p = model.map.pois[aid]
                ctrl.prompt_text = p.loop.prompt if (p.loop and p.loop.prompt) else f"(No prompt yet for {p.name}.)"
            ctrl.show_prompt = True

    def _enter_mixer(self):
        self.context.controller.change_state("mixer")

    def _enter_dance_mode(self):
        audio = self.context.audio
        model = self.context.model
        k, i = audio.active_source
        loop, title, spec = None, "", None
        if k == "zone":
            zr = model.map.zones[i]
            loop, title, spec = zr.loop, zr.spec.name, zr.spec
        else:
            p = model.map.pois[i]
            loop, title, spec = p.loop, p.name, model.map.zones[p.zone_id].spec
        
        if loop:
            self.context.controller.change_state("dance", loop=loop, src_title=title, zone_spec=spec)
        else:
            print("[DANCE] No loop ready yet for current source.")

    def _handle_interaction(self):
        ctrl = self.context.controller
        model = self.context.model
        px, py = model.player.tile_x, model.player.tile_y
        q_pid = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pid = model.map.pois_at.get((px + dx, py + dy))
                if pid and model.map.pois[pid].role == "quest_giver":
                    q_pid = pid
                    break
            if q_pid: break

        if q_pid:
            if model.active_quest is None:
                npcs = [p for p in model.map.pois.values() if p.kind == "npc" and p.name != "Boss Skuggs"]
                if npcs:
                    target = random.choice(npcs)
                    tz = model.map.zones[target.zone_id]
                    model.active_quest = Quest(
                        giver_pid=q_pid, target_pid=target.pid, target_name=target.name,
                        target_tile=target.tile, target_zone=target.zone_id,
                        target_zone_name=tz.spec.name, accepted=True,
                    )
            if model.active_quest:
                q = model.active_quest
                ctrl.quest_text = (
                    f"Find the alien '{q.target_name}'.\n"
                    f"Location: {q.target_zone_name} at tile {q.target_tile}.\n\n"
                    f"Tip: stand next to them to hear their tune."
                )
                ctrl.show_quest = True
        elif ctrl.show_quest:
            ctrl.show_quest = False

    def _regenerate_zone_audio(self):
        zid = self.context.model.current_zone_id
        self.context.model.map.zones[zid].loop = None
        self.context.audio.request_zone(zid, priority=(0, 0, 0), force=True)

    def _cycle_mood(self):
        self.context.controller.cycle_mood()
        self._regenerate_zone_audio()

    def _edit_mood(self):
        self.context.controller.edit_mood_text()
        self._regenerate_zone_audio()

    def _toggle_record(self):
        audio = self.context.audio
        if audio.recorder.is_recording():
            audio.recorder.stop()
            audio.record_armed = False
            print("[REC] stopped.")
        else:
            audio.record_armed = True
            print("[REC] armed: will start at next loop boundary.")

    def _toggle_listen_mode(self):
        self.context.audio.toggle_listen_mode()
        print("[Mix] mode:", self.context.audio.listen_mode)

    def _toggle_broadcast(self):
        audio = self.context.audio
        player = self.context.model.player
        if player.broadcasting:
            audio.stop_broadcast()
            player.broadcasting = False
        else:
            if player.broadcast_index is not None and 0 <= player.broadcast_index < len(player.inventory_songs):
                song = player.inventory_songs[player.broadcast_index]
                audio.start_broadcast(song)
                player.broadcasting = True

    def _stop_broadcast(self):
        self.context.audio.stop_broadcast()
        self.context.model.player.broadcasting = False

    def _select_broadcast_slot(self, slot: int):
        player = self.context.model.player
        if slot < len(player.inventory_songs):
            player.broadcast_index = slot
            if player.broadcasting:
                self.context.audio.start_broadcast(player.inventory_songs[slot])

    def _adjust_broadcast_volume(self, delta: float):
        v = self.context.audio.broadcast.volume + delta
        self.context.audio.broadcast.set_volume(v)

    def _reload_promptgen_and_regen(self):
        import importlib
        from cosmic_underground import promptgen as _pg
        importlib.reload(_pg)
        print(f"[PromptGen] Reloaded {getattr(_pg, 'SCHEMA_VERSION', '?')}")
        k, i = self.context.audio.active_source
        if k == "zone":
            self.context.model.map.zones[i].loop = None
            self.context.audio.request_zone(i, priority=(0, 0, 0), force=True)
        else:
            self.context.model.map.pois[i].loop = None
            self.context.audio.request_poi(i, priority=(0, 0, 0), force=True)

    def update(self, dt: int):
        ctrl = self.context.controller
        model = self.context.model
        audio = self.context.audio

        # --- Movement ---
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        dy = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - (keys[pygame.K_UP] or keys[pygame.K_w])
        if dx or dy:
            model.move_player(dx * model.player.speed, dy * model.player.speed)
            q = ctrl.active_quest
            if q:
                px, py = model.player.tile_x, model.player.tile_y
                tx, ty = q.target_tile
                if max(abs(px - tx), abs(py - ty)) <= 1:
                    ctrl.quest_text = f"Quest complete! You found {q.target_name} in {q.target_zone_name}."
                    ctrl.show_quest = True
                    ctrl.active_quest = None
                    model.quest_completed = True
        
        # --- Influence Update ---
        dt_sec = dt / 1000.0
        if getattr(audio, "broadcast", None):
            audio.broadcast.apply_influence(world_model=model, dt_sec=dt_sec)

    def draw(self, screen: pygame.Surface):
        ctrl = self.context.controller
        audio = self.context.audio
        view = self.context.view

        # Overworld draw
        view.draw(
            screen,
            audio.record_armed,
            audio.recorder.is_recording(),
            ctrl.show_prompt,
            ctrl.prompt_text,
            ctrl.show_quest,
            ctrl.quest_text,               
        )

        # draw top-right music widget
        view.draw_music_widget(screen, audio)
