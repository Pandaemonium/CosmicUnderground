from typing import Any, Tuple
import secrets
from cosmic_underground.core.config import *
from cosmic_underground.core.mapgen import RegionMap
from cosmic_underground.core.models import Player

# "I love dancing with my alien friends"

class WorldModel:
    def __init__(self):
        self.world_seed = secrets.randbits(32)
        self.map = RegionMap(MAP_W, MAP_H, self.world_seed)

        sx, sy = START_TILE
        self.player = Player(sx, sy, sx*TILE_W + TILE_W/2, sy*TILE_H + TILE_H/2, speed=6.0)
        self.current_zone_id = self.map.zone_of[sx][sy]

        # runtime flags/context
        self.time_of_day = "night"
        self.weather = None
        self.heat = 0.15
        self.debt_pressure = 0.4
        self.festival = False
        self.cosmic_event = False

        # listeners
        self._tile_listeners = []
        self._zone_listeners: List[Any] = []
        
        self.quest: Optional[Quest] = None
        self.active_quest: Optional[Quest] = None
        self.quest_giver_pid = getattr(self.map, "quest_giver_pid", None)
        self.quest_completed = False
        
        def add_tile_changed_listener(self, fn):
            self._tile_listeners.append(fn)
        def move_player(self, dx, dy):
            old_tx, old_ty = self.player.tile_x, self.player.tile_y
            self.player.px += dx; self.player.py += dy
            self.player.tile_x = max(0, min(MAP_W-1, int(self.player.px // TILE_W)))
            self.player.tile_y = max(0, min(MAP_H-1, int(self.player.py // TILE_H)))
            if (self.player.tile_x, self.player.tile_y) != (old_tx, old_ty):
                for fn in self._tile_listeners: fn((old_tx,old_ty), (self.player.tile_x, self.player.tile_y))

    def add_tile_changed_listener(self, fn): self._tile_listeners.append(fn)
    def add_zone_changed_listener(self, fn): self._zone_listeners.append(fn)

    def move_player(self, dx: float, dy: float):
        # pixel move
        self.player.px += dx; self.player.py += dy
        # clamp into map
        self.player.px = max(0, min(self.player.px, MAP_W*TILE_W-1))
        self.player.py = max(0, min(self.player.py, MAP_H*TILE_H-1))
        # tile
        nx = int(self.player.px // TILE_W)
        ny = int(self.player.py // TILE_H)
        if (nx,ny) != (self.player.tile_x, self.player.tile_y):
            oldt = (self.player.tile_x, self.player.tile_y)
            self.player.tile_x, self.player.tile_y = nx, ny
            for fn in self._tile_listeners: fn(oldt, (nx,ny))

            zid = self.map.zone_of[nx][ny]
            if zid != self.current_zone_id:
                oldz = self.current_zone_id; self.current_zone_id = zid
                for fn in self._zone_listeners: fn(oldz, zid)