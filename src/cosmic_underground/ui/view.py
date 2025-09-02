import math, pygame
from typing import Dict, List, Tuple, Optional
from cosmic_underground.core.config import TILE_W, TILE_H
from cosmic_underground.core.models import POI, WorldModel
from cosmic_underground.audio import AudioService

class GameView:
    def __init__(self, model: WorldModel, audio: AudioService):
        self.m = model
        self.audio = audio
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        
        # --- NEW: zone title fonts (per-biome), cache, and sizing
        self.zone_font_cache: dict[tuple[str,int], pygame.font.Font] = {}
        self.zone_name_px = 30  # size for big zone names (tweak to taste)
        
        # Pick families that usually exist on Windows/macOS/Linux; will fallback if missing.
        self.biome_font_family = {
            "Crystal Canyons": "georgia",       # serif, crystalline vibe
            "Slime Caves": "tahoma",            # clean sans
            "Centerspire": "consolas",          # techy mono
            "Groove Pits": "verdana",           # chunky groove
            "Polar Ice": "trebuchetms",         # airy sans
            "Mag-Lev Yards": "couriernew",      # industrial mono
            "Electro Marsh": "segoe ui",        # modern sans
        }
        
        self.player_img = self._load_sprite(PLAYER_SPRITE, target_h=48)
        self.player_img_complete = self._load_sprite(PLAYER_SPRITE_COMPLETE, target_h=48)
        
        # If a family isn’t found, SysFont will pick “best available”.
    def _get_biome_font(self, biome: str, size: int) -> pygame.font.Font:
        key = (biome, size)
        if key in self.zone_font_cache:
            return self.zone_font_cache[key]
        family = self.biome_font_family.get(biome, "trebuchetms")
        try:
            font = pygame.font.SysFont(family, size, bold=True)
        except Exception:
            font = pygame.font.SysFont(None, size, bold=True)
        self.zone_font_cache[key] = font
        return font
    
    def _blit_text_outline(self, surf: pygame.Surface, text: str, font: pygame.font.Font,
                           x: int, y: int, fill=(240,240,255), outline=(255,255,255), px: int = 2):
        # Draw an outline by rendering the text offset in 8 directions, then fill on top
        txt = font.render(text, True, fill)
        out = font.render(text, True, outline)
        for ox, oy in ((-px,0),(px,0),(0,-px),(0,px),(-px,-px),(px,-px),(-px,px),(px,px)):
            surf.blit(out, (x+ox, y+oy))
        surf.blit(txt, (x, y))

    def _draw_glow(self, surf: pygame.Surface, rect: pygame.Rect, pulse_t: float):
        glow_color = (170 + int(60*(0.5+0.5*math.sin(pulse_t*2.0))), 60, 220)
        glow = pygame.Surface((rect.w+16, rect.h+16), pygame.SRCALPHA)
        base = pygame.Rect(8, 8, rect.w, rect.h)
        for i, alpha in enumerate((90, 60, 30)):
            pygame.draw.rect(glow, (*glow_color, alpha), base.inflate(8+i*6, 8+i*6), width=3, border_radius=14)
        surf.blit(glow, (rect.x-8, rect.y-8))
    
    def _draw_beacon_glow(self, surf: pygame.Surface, cx: int, cy: int, t: float):
        """Strong pulsating laser-blue glow for the quest giver."""
        # laser blue
        base = (80, 200, 255)
        # fast pulse
        s = (math.sin(t*3.2) * 0.5 + 0.5)  # 0..1
        # 3 concentric rings + soft disc
        for i, alpha in enumerate((180, 120, 70)):
            r = int(22 + i*10 + s*8)
            pygame.draw.circle(surf, (*base, alpha), (cx, cy), r, width=3)
        # soft inner disc
        inner = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(inner, (*base, 80 + int(70*s)), (30, 30), 24)
        surf.blit(inner, (cx-30, cy-30), special_flags=pygame.BLEND_PREMULTIPLIED)

    def draw(self, screen: pygame.Surface, record_armed: bool, recorder_active: bool,
         show_prompt: bool=False, prompt_text: str="",
         show_quest: bool=False, quest_text: str=""):

        screen.fill((14,10,18))
        pulse_t = pygame.time.get_ticks()/1000.0
        W, H = screen.get_size()
        cam_x = self.m.player.px - W/2
        cam_y = self.m.player.py - H/2

        min_zx = max(0, int(cam_x // TILE_W) - 1)
        max_zx = min(MAP_W-1, int((cam_x + W) // TILE_W) + 1)
        min_zy = max(0, int(cam_y // TILE_H) - 1)
        max_zy = min(MAP_H-1, int((cam_y + H) // TILE_H) + 1)

        active_k, active_id = self.audio.active_source
        active_tile = None
        if active_k == "zone":
            # pick nearest tile of that zone to player (approx: use player's tile)
            active_tile = (self.m.player.tile_x, self.m.player.tile_y)
        else:
            active_tile = self.m.map.pois[active_id].tile

        # Draw tiles
        # Build visible tiles grouped by zone id
        visible_by_zone: Dict[int, List[Tuple[int,int]]] = {}
        visible_pois: List[Tuple[POI, int, int]] = []  # (poi, cx, cy)
        
        for zy in range(min_zy, max_zy+1):
            for zx in range(min_zx, max_zx+1):
                zid = self.m.map.zone_of[zx][zy]
                zr = self.m.map.zones[zid]
                sx = int(zx*TILE_W - cam_x); sy = int(zy*TILE_H - cam_y)
                rect = pygame.Rect(sx+2, sy+2, TILE_W-4, TILE_H-4)
        
                # base tile color per zone
                base_col = self.m.map.zone_color[zid]
                if (zx,zy) == (self.m.player.tile_x, self.m.player.tile_y):
                    base_col = (70, 88, 140)
                pygame.draw.rect(screen, base_col, rect, border_radius=12)
        
                # audible source marker ring
                if active_tile == (zx,zy):
                    pygame.draw.rect(screen, (200,120,255), rect, width=3, border_radius=12)
        
                # collect per-zone tiles
                visible_by_zone.setdefault(zid, []).append((zx, zy))
        
                # collect POIs on this tile (for post-pass)
                pid = self.m.map.pois_at.get((zx,zy))
                if pid:
                    poi = self.m.map.pois[pid]
                    cx = rect.centerx; cy = rect.centery
                    visible_pois.append((poi, cx, cy))
                    # simple marker (we'll do glow in overlay pass)
                    color = (255,220,90) if poi.name=="Boss Skuggs" else ((200,255,200) if poi.kind=="npc" else (200,200,255))
                    pygame.draw.circle(screen, color, (cx, cy), 10)
        
        # --- 2) FX overlay (zone perimeter + beacon glow), then blit once ---
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        
        # zone perimeters (only for ready zones)
        for zid, tiles in visible_by_zone.items():
            zr = self.m.map.zones[zid]
            if zr.loop and not zr.error:
                self._stroke_zone_edges(overlay, tiles, cam_x, cam_y, pulse_t)
        
        # quest beacon strong laser-blue glow
        for poi, cx, cy in visible_pois:
            if poi.role == "quest_giver":
                self._draw_beacon_glow(overlay, cx, cy, pulse_t)
        
        screen.blit(overlay, (0, 0))
        
        # --- NEW: draw BIG zone names (outlined) after overlay, so they sit above tiles
        for zid, tiles in visible_by_zone.items():
            ax, ay = self.m.map.zone_anchor[zid]
            if not (min_zx <= ax <= max_zx and min_zy <= ay <= max_zy):
                continue  # anchor not visible; skip
            px = int(ax*TILE_W - cam_x) + 12
            py = int(ay*TILE_H - cam_y) + 10
            self._blit_label(screen, self.m.map.zones[zid].spec.name, px, py)

        
        # --- 4) POI name labels (centered UNDER marker) ---
        for poi, cx, cy in visible_pois:
            text_w, _ = self.small.size(poi.name)
            pad = 4
            x = int(cx - (text_w + pad*2)//2)
            y = int(cy + 12)  # below the circle (radius ~10) with small gap
            self._blit_label(screen, poi.name, x, y)

        
        

        # player (sprite)
        px = int(self.m.player.px - cam_x)
        py = int(self.m.player.py - cam_y)
        
        img = self.player_img_complete if getattr(self.m, "quest_completed", False) else self.player_img
        if img:
            # center sprite on player position
            rect = img.get_rect(center=(px, py))
            screen.blit(img, rect.topleft)
        else:
            # fallback if sprite failed to load
            pygame.draw.circle(screen, (255,100,120), (px, py), 10)

        # HUD
        hud = ["Move: WASD/Arrows | F interact | G regen zone | M cycle mood | E edit mood | N panel | P prompt | R record | Esc quit"]

        if record_armed:    hud.append("REC ARMED: starts at next loop boundary.")
        if recorder_active: hud.append("RECORDING… R to stop (auto-stops at boundary).")
        y = H - 22*len(hud) - 8
        for line in hud:
            screen.blit(self.font.render(line, True, (240,240,240)), (10, y)); y += 22
        
        # Quest status line
        q = getattr(self.m, "active_quest", None)
        if q:
            hud.append(f"QUEST: Find {q.target_name} in {q.target_zone_name}")

        # Prompt overlay
        if show_prompt and prompt_text:
            # wrap
            max_w = int(W * 0.8)
            pad   = 14
            title_font = pygame.font.SysFont("consolas", 20, bold=True)
            body_font  = pygame.font.SysFont("consolas", 18)
            title = "Audio Prompt"
            lines = self._wrap_text(prompt_text, body_font, max_w)

            title_w, title_h = title_font.size(title)
            body_h = sum(body_font.size(line)[1] + 4 for line in lines)
            box_w = min(max_w, max(title_w, max((body_font.size(l)[0] for l in lines), default=0))) + pad*2
            box_h = title_h + 10 + body_h + pad*2

            box_x = 10
            box_y = H - box_h - 10

            surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(surf, (20, 16, 28, 220), (0, 0, box_w, box_h), border_radius=10)
            pygame.draw.rect(surf, (180, 120, 255, 255), (0, 0, box_w, box_h), width=2, border_radius=10)
            screen.blit(surf, (box_x, box_y))

            screen.blit(title_font.render(title, True, (240, 230, 255)), (box_x + pad, box_y + pad))
            yy = box_y + pad + title_h + 6
            for line in lines:
                screen.blit(body_font.render(line, True, (230, 230, 240)), (box_x + pad, yy))
                yy += body_font.size(line)[1] + 4
        # Quest overlay
        if show_quest and quest_text:
            max_w = int(W * 0.8)
            pad   = 14
            title_font = pygame.font.SysFont("consolas", 20, bold=True)
            body_font  = pygame.font.SysFont("consolas", 18)
            title = "Quest"
            lines = self._wrap_text(quest_text, body_font, max_w)
        
            title_w, title_h = title_font.size(title)
            body_h = sum(body_font.size(line)[1] + 4 for line in lines)
            box_w = min(max_w, max(title_w, max((body_font.size(l)[0] for l in lines), default=0))) + pad*2
            box_h = title_h + 10 + body_h + pad*2
        
            box_x = 10
            box_y = H - box_h - 10
        
            surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(surf, (16, 20, 30, 235), (0, 0, box_w, box_h), border_radius=10)
            pygame.draw.rect(surf, (90, 190, 255, 255), (0, 0, box_w, box_h), width=2, border_radius=10)
            screen.blit(surf, (box_x, box_y))
        
            screen.blit(title_font.render(title, True, (220, 240, 255)), (box_x + pad, box_y + pad))
            yy = box_y + pad + title_h + 6
            for line in lines:
                screen.blit(body_font.render(line, True, (225, 235, 245)), (box_x + pad, yy))
                yy += body_font.size(line)[1] + 4


    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = w if not cur else cur + " " + w
            if font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines
    
    def _load_sprite(self, path: str, target_h: int = 48) -> Optional[pygame.Surface]:
        """Load a sprite with alpha, scale to target height preserving aspect. Returns None if load fails."""
        try:
            img = pygame.image.load(path).convert_alpha()
            w, h = img.get_size()
            if h != target_h:
                scale = target_h / float(h)
                img = pygame.transform.smoothscale(img, (int(w*scale), target_h))
            return img
        except Exception as e:
            print(f"[Sprite] load failed: {path} ({e})")
            return None

    
    def _blit_label(self, screen: pygame.Surface, text: str, x: int, y: int,
                bgcolor=(20,16,28), fg=(240,240,255)):
        surf = self.small.render(text, True, fg)
        pad = 4
        box = pygame.Surface((surf.get_width()+pad*2, surf.get_height()+pad*2), pygame.SRCALPHA)
        pygame.draw.rect(box, (*bgcolor, 210), box.get_rect(), border_radius=6)
        pygame.draw.rect(box, (180,120,255), box.get_rect(), width=1, border_radius=6)
        box.blit(surf, (pad, pad))
        screen.blit(box, (x, y))
    
    def _stroke_zone_edges(self, overlay: pygame.Surface, tiles: List[Tuple[int,int]],
                           cam_x: float, cam_y: float, t: float):
        S = set(tiles)
        base = (170 + int(60*(0.5 + 0.5*math.sin(t*2.0))), 60, 220)
        # draw thick→thin on the SAME overlay
        for width, alpha in ((8,70), (5,110), (2,180)):
            col = (*base, alpha)
            for (tx, ty) in tiles:
                rx = int(tx*TILE_W - cam_x) + 2
                ry = int(ty*TILE_H - cam_y) + 2
                rw = TILE_W - 4; rh = TILE_H - 4
                if (tx, ty-1) not in S: pygame.draw.line(overlay, col, (rx, ry), (rx+rw, ry), width)
                if (tx+1, ty) not in S: pygame.draw.line(overlay, col, (rx+rw, ry), (rx+rw, ry+rh), width)
                if (tx, ty+1) not in S: pygame.draw.line(overlay, col, (rx, ry+rh), (rx+rw, ry+rh), width)
                if (tx-1, ty) not in S: pygame.draw.line(overlay, col, (rx, ry), (rx, ry+rh), width)
