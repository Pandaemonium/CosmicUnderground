#!/usr/bin/env python3
# Alien DJ – Overworld zones + POIs, priority audio (2 workers), immediate crossfade
# Requirements: pygame, numpy, soundfile, torch, stable_audio_tools, promptgen.py in same folder

import os, sys
ROOT = os.path.dirname(__file__)
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


import pygame, random

from cosmic_underground.minigames.dance.engine import DanceMinigame
from cosmic_underground.minigames.dance import config as D

from cosmic_underground.core import config as C
from cosmic_underground.core.models import ZoneSpec, GeneratedLoop, ZoneRuntime, POI, Player, Quest
from cosmic_underground.core.world import WorldModel
from cosmic_underground.audio.service import AudioService
from cosmic_underground.ui.view import GameView


# --- Display/config ---
DEFAULT_FULLSCREEN = True
SCREEN_W, SCREEN_H = 1200, 700
FPS = 60

ENGINE_SR = 44100
      # huge finite map
ZONE_MIN, ZONE_MAX = 10, 40         # contiguous blob size per zone
AVG_ZONE = 20                       # target avg (used for seed count)
POIS_NPC_RANGE = (0, 2)
POIS_OBJ_RANGE = (0, 1)

START_ZONE_NAME  = "Scrapyard Funk"
START_THEME_WAV  = r"C:\Games\CosmicUnderground\inventory\rec_1756545018_Scrapyard Funk_d5ae11.wav"

PLAYER_SPRITE = r"C:\Games\CosmicUnderground\sprites\character1.png"
PLAYER_SPRITE_COMPLETE = "C:\Games\CosmicUnderground\sprites\laser_bunny.png"  # swap to a different file later if you like

# Generation/caching
MAX_ACTIVE_LOOPS = 120             # LRU cap
GEN_WORKERS = 1                    # concurrent Stable Audio generations





# ======= Prompt helpers (light biasing without changing your promptgen) =======
def tokens_for_poi(poi: POI) -> List[str]:
    """Optional prepend tokens to bias short prompts without changing promptgen."""
    if poi.name == "Boss Skuggs":
        return ["cosmic", "talkbox", "boogie"]  # signature
    if poi.kind == "npc":
        return ["alien", "funk", "bass"]        # lean funkier
    if poi.kind == "object":
        return ["weird", "motif"]               # sparser
    return []


# ======= Controller =======
class GameController:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ – Overworld")
        self.fullscreen = C.DEFAULT_FULLSCREEN
        self.screen = self._apply_display_mode()
        self.clock  = pygame.time.Clock()

        self.model = WorldModel()
        self.audio = AudioService(self.model)
        self.view  = GameView(self.model, self.audio)
        
        self.mode = "overworld"
        self.dance = DanceMinigame(on_finish=self._on_dance_finish)

        self.show_prompt = False
        self.prompt_text = ""
        self.show_panel = False  # toggleable Now Playing (stub – we keep prompt for now)

        # Auto-hide prompt when tile changes (optional)
        self.model.add_tile_changed_listener(lambda oldt, newt: setattr(self, "show_prompt", False))
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
        q = self.active_quest  # uses the bridge, equivalent to self.model.active_quest
    
        if res.passed and q:
            # reward: swap sprite (ignore if you don’t have this helper)
            try:
                self._set_player_sprite(PLAYER_SPRITE_COMPLETE)
            except Exception as e:
                print("[Sprite] swap failed:", e)
    
            self.quest_text = f"Quest complete! You danced with {q.target_name} in {q.target_zone_name}."
            self.show_quest = True
            self.active_quest = None  # clear it

    
    
    def _adjacent_pois(self) -> List[POI]:
        px, py = self.model.player.tile_x, self.model.player.tile_y
        out = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                tx, ty = px+dx, py+dy
                if 0 <= tx < C.MAP_W and 0 <= ty < C.MAP_H:
                    pid = self.model.map.pois_at.get((tx,ty))
                    if pid:
                        out.append(self.model.map.pois[pid])
        return out
    

    

    def _apply_display_mode(self):
        global SCREEN_W, SCREEN_H
        if self.fullscreen:
            screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode((1200,700), pygame.RESIZABLE)
        SCREEN_W, SCREEN_H = screen.get_size()
        return screen

    def cycle_mood(self):
        order = ["calm","energetic","angry","triumphant","melancholy","playful","brooding","gritty","glittery","funky"]
        zid = self.model.current_zone_id
        z = self.model.map.zones[zid]
        cur = z.spec.mood.lower()
        z.spec.mood = order[(order.index(cur)+1) % len(order)] if cur in order else "calm"

    def edit_mood_text(self):
        pygame.key.set_repeat(0)
        font = pygame.font.SysFont("consolas", 20)
        entered = ""; done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN: done = True
                    elif e.key == pygame.K_ESCAPE: done = True
                    elif e.key == pygame.K_BACKSPACE: entered = entered[:-1]
                    else:
                        if e.unicode and 32 <= ord(e.unicode) < 127: entered += e.unicode
            self.screen.fill((10,10,15))
            self.screen.blit(font.render("Enter mood/descriptor (Enter=OK, Esc=Cancel):", True, (240,240,240)), (40, SCREEN_H//2 - 30))
            self.screen.blit(font.render(entered, True, (180,255,180)), (40, SCREEN_H//2 + 10))
            pygame.display.flip(); self.clock.tick(30)
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
    
                # ---- DANCE MODE: consume events and never fall through ----
                if self.mode == "dance":
                    # pass relevant events to the minigame
                    if e.type in (pygame.KEYDOWN, pygame.KEYUP, self.audio.player.boundary_event):
                        if self.dance:
                            self.dance.handle_event(e)
    
                    # ESC/Q exits dance mode ONLY
                    if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                        # let the minigame finalize if it wants
                        if self.dance:
                            self.dance.handle_event(e)
                        self.mode = "world"   # or "overworld" — be consistent everywhere
                    # IMPORTANT: swallow event so overworld handlers don't see this ESC
                    continue
    
                # ---- OVERWORLD (only runs when not in dance) ----
                if e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    global SCREEN_W, SCREEN_H
                    SCREEN_W, SCREEN_H = e.w, e.h
                    self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    
                elif e.type == self.audio.player.boundary_event:
                    self.audio.on_boundary_tick()
    
                if e.type == pygame.KEYDOWN:
                    # ESC/Q quits game only in overworld
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
                        import importlib, promptgen as _pg
                        importlib.reload(_pg)
                        print(f"[PromptGen] Reloaded {getattr(_pg, 'SCHEMA_VERSION', '?')}")
                        k, i = self.audio.active_source
                        if k=="zone":
                            self.model.map.zones[i].loop = None
                            self.audio.request_zone(i, priority=(0,0,0), force=True)
                        else:
                            self.model.map.pois[i].loop = None
                            self.audio.request_poi(i, priority=(0,0,0), force=True)
                            
                    elif e.key == pygame.K_c and self.mode == "dance":
                        # toggle calibration overlay
                        if self.dance:
                            self.dance.handle_event(e)
                        continue
    
                    elif e.key == pygame.K_k:
                        # Start dance mode on the current audible source if its loop is ready
                        k, i = self.audio.active_source
                        loop = None; title = ""; spec = None
                        if k == "zone":
                            zr = self.model.map.zones[i]
                            loop = zr.loop; title = zr.spec.name; spec = zr.spec
                        else:
                            p = self.model.map.pois[i]
                            loop = p.loop; title = p.name; spec = self.model.map.zones[p.zone_id].spec
                        if loop:
                            # IMPORTANT: stop overworld playback so dance owns mixer.music
                            try:
                                self.audio.player.stop(fade_ms=0)
                            except Exception:
                                pass
                            self.mode = "dance"
                            self.dance.start_for_loop(loop=loop, src_title=title, player=self.audio.player, zone_spec=spec)
                        else:
                            print("[DANCE] No loop ready yet for current source.")

                    elif e.key == pygame.K_f:
                        # Interact: if adjacent to quest giver, pop quest card (create quest if needed)
                        px, py = self.model.player.tile_x, self.model.player.tile_y
                        # find any POI within 1-tile Chebyshev radius that's a quest giver
                        q_pid = None
                        for dx in (-1,0,1):
                            for dy in (-1,0,1):
                                tx, ty = px+dx, py+dy
                                pid = self.model.map.pois_at.get((tx,ty))
                                if not pid: continue
                                poi = self.model.map.pois[pid]
                                if poi.role == "quest_giver":
                                    q_pid = pid
                                    break
                            if q_pid: break
                    
                        if q_pid:
                            # if we don't already have a quest, pick a target NPC anywhere (not Boss Skuggs)
                            if self.model.active_quest is None:
                                # pick a target npc
                                npcs = [p for p in self.model.map.pois.values() if p.kind=="npc" and p.name != "Boss Skuggs"]
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
                                        accepted=True
                                    )
                            # show card (existing or just-created)
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
                            # Optional: click F again to dismiss the quest box if it's open
                            if self.show_quest:
                                self.show_quest = False

                    elif e.key == pygame.K_g:
                        # Regenerate current zone
                        zid = self.model.current_zone_id
                        self.model.map.zones[zid].loop = None
                        self.audio.request_zone(zid, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_m:
                        self.cycle_mood()
                        self.audio.request_zone(self.model.current_zone_id, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_e:
                        self.edit_mood_text()
                        self.audio.request_zone(self.model.current_zone_id, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_r:
                        if self.audio.recorder.is_recording():
                            self.audio.recorder.stop(); self.audio.record_armed = False
                            print("[REC] stopped.")
                        else:
                            self.audio.record_armed = True
                            print("[REC] armed: will start at next loop boundary.")
                    elif e.key == pygame.K_i:
                        print_inventory("./inventory")

            # --- Movement only in overworld ---
            if self.mode != "dance":
                keys = pygame.key.get_pressed()
                dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
                dy = (keys[pygame.K_DOWN]  or keys[pygame.K_s]) - (keys[pygame.K_UP]   or keys[pygame.K_w])
                if dx or dy:
                    self.model.move_player(dx*self.model.player.speed, dy*self.model.player.speed)
                    # Quest auto-complete if adjacent to target
                    # --- quest completion check ---
                    q = self.model.active_quest
                    if q:
                        px, py = self.model.player.tile_x, self.model.player.tile_y
                        tx, ty = q.target_tile
                        if max(abs(px - tx), abs(py - ty)) <= 1:
                            # Completed!
                            self.quest_text = f"Quest complete! You found {q.target_name} in {q.target_zone_name}."
                            self.show_quest = True  # reuse the quest modal to announce completion
                            self.model.active_quest = None
                            self.model.quest_completed = True
            
            # 2) Dance-mode frame branch: update & draw, then continue the while-loop
            if self.mode == "dance":
                dt = self.clock.get_time()
                self.dance.update(dt)
                self.screen.fill((10, 10, 14))
                self.dance.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(FPS)
                continue  # skip overworld draw this frame


            self.view.draw(self.screen,
               self.audio.record_armed,
               self.audio.recorder.is_recording(),
               self.show_prompt, self.prompt_text,
               self.show_quest, self.quest_text)

            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

# ======= utils =======
def print_inventory(inv_dir: str):
    try:
        files = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)"); return
    print("[INV] Recorded clips:")
    for f in files: print("  -", f)

# ======= entry =======
if __name__ == "__main__":
    try:
        GameController().run()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
