import os, threading, itertools, heapq
import soundfile as sf
from typing import Optional, Tuple, Dict, List, Any
import random

from cosmic_underground.core.config import (
    START_TILE, START_THEME_WAV, MAX_ACTIVE_LOOPS, GEN_WORKERS,
    MAP_W, MAP_H
)
from cosmic_underground.core.models import GeneratedLoop, ZoneSpec, WorldModel, POI
from cosmic_underground.audio.provider import LocalStableAudioProvider
from cosmic_underground.audio.player import AudioPlayer
from cosmic_underground.audio.recorder import Recorder

class AudioService:
    """
    Priority tiers recomputed on tile change:
      0 = current audible source (zone or poi)
      1 = POIs within radius 1 of the player
      2 = zone neighbors in view
      3 = backlog
    """
    def __init__(self, model: WorldModel):
        self.m = model
        # context → promptgen
        def _ctx():
            return dict(
                time_of_day=self.m.time_of_day,
                weather=self.m.weather,
                heat=self.m.heat,
                debt_pressure=self.m.debt_pressure,
                festival=self.m.festival,
                cosmic_event=self.m.cosmic_event,
            )
        self.provider = LocalStableAudioProvider(context_fn=_ctx)
        self.player = AudioPlayer()
        self.recorder = Recorder()
        self.record_armed = False
        self.auto_stop_at_end = True

        # Task queue
        self._heap = []  # (prio_tuple, seq, task_dict)
        self._counter = itertools.count()
        self._pending: Dict[Tuple[str,int], int] = {}  # (kind,id) -> latest token
        self._lock = threading.Lock()

        # Active source
        self.active_source: Tuple[str,int] = ("zone", self.m.current_zone_id)
        


        # initialize once at startup
        self.active_source = self._current_active_source()
        rt0 = self._get_runtime(self.active_source)
        if rt0.loop:
            self.player.play_loop(rt0.loop.wav_path, rt0.loop.duration_sec, fade_ms=140)
        else:
            # try preload for start zone, fall back to generate
            if not self._maybe_preload_start_zone():
                self.request_generate(self.active_source, priority=(0,0), force=True)
        self._reprioritize_all()
        
        # now start the worker (provider must already exist)
        self._worker_stop = False
        self._workers = []
        for _ in range(max(1, GEN_WORKERS)):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)


        # listeners
        self.m.add_tile_changed_listener(self.on_tile_changed)
        self.m.add_zone_changed_listener(self.on_zone_changed)

        # preload theme for start zone
        self._maybe_preload_zone(self.m.current_zone_id)
        # ensure initial audible source is playing
        self._ensure_audible_playing()

        # prefetch immediate neighbors
        self._reprioritize_all()

    # ---------- audible source selection ----------
    @staticmethod
    def _chebyshev(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def _nearby_pois(self) -> List[POI]:
        px, py = self.m.player.tile_x, self.m.player.tile_y
        out = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                tx, ty = px+dx, py+dy
                if not (0 <= tx < MAP_W and 0 <= ty < MAP_H): continue
                pid = self.m.map.pois_at.get((tx,ty))
                if pid: out.append(self.m.map.pois[pid])
        # Order by distance, then kind preference (npc over object), then rarity desc
        out.sort(key=lambda p: (abs(p.tile[0]-px)+abs(p.tile[1]-py), 0 if p.kind=="npc" else 1, -p.rarity))
        return out

    def _pick_audible(self) -> Tuple[str,int]:
        cands = self._nearby_pois()
        if cands:
            return ("poi", cands[0].pid)
        return ("zone", self.m.current_zone_id)

    def _ensure_audible_playing(self):
        kind, idv = self.active_source
        if kind == "zone":
            zr = self.m.map.zones[idv]
            if zr.loop:
                self.player.play_loop(zr.loop.wav_path, zr.loop.duration_sec, cross_ms=220)
            else:
                self.request_zone(idv, priority=(0,0), force=True)
        else:
            poi = self.m.map.pois[idv]
            if poi.loop:
                self.player.play_loop(poi.loop.wav_path, poi.loop.duration_sec, cross_ms=220)
            else:
                self.request_poi(idv, priority=(0,0), force=True)

    # ---------- tasks ----------
    def request_zone(self, zid: int, priority: Optional[Tuple[int,int]]=None, force: bool=False):
        prio = priority or self._priority_for(("zone", zid))
        self.request_generate(("zone", zid), priority=prio, force=force)

    def request_poi(self, pid: int, priority: Optional[Tuple[int,int]]=None, force: bool=False):
        prio = priority or self._priority_for(("poi", pid))
        self.request_generate(("poi", pid), priority=prio, force=force)

    def _pop_task(self):
        with self._lock:
            while self._heap:
                prio, token, src = heapq.heappop(self._heap)
                if self._pending.get(src) != token:
                    continue  # stale
                return prio, token, src
        return None

    # ---------- worker ----------
    def _worker_loop(self):
        import time
        while not self._worker_stop:
            item = self._pop_task()
            if item is None:
                time.sleep(0.01); continue
            _, _, src = item
            rt = self._get_runtime(src)
            if rt.generating or rt.loop is not None:
                continue
            rt.generating = True; rt.error = None
            try:
                loop = self._generate_for(src)
                rt.loop = loop
                # if the thing we just generated is the *currently audible* source, play it now
                if src == self.active_source:
                    self.player.play_loop(loop.wav_path, loop.duration_sec, fade_ms=200)
            except Exception as e:
                rt.error = str(e)
                print(f"[FATAL][GEN] {src}: {e}", flush=True)
            finally:
                rt.generating = False
            self._prune_cache()

    # ---------- priority ----------
    def _priority_for_zone(self, zid: int) -> Tuple[int,int,int]:
        """(tier, manhattan_dist_to_centroid, -recency) — here recency omitted for simplicity."""
        px, py = self.m.player.tile_x, self.m.player.tile_y
        kind, aid = self.active_source
        tier = 3
        if kind == "zone" and aid == zid: tier = 0
        cx, cy = self.m.map.zones[zid].centroid
        dist = int(abs(px - cx) + abs(py - cy))
        # neighbors (by centroid approx) get tier 2
        if tier != 0 and dist <= 4: tier = 2
        return (tier, dist, 0)

    def _priority_for_poi(self, pid: int) -> Tuple[int,int,int]:
        px, py = self.m.player.tile_x, self.m.player.tile_y
        poi = self.m.map.pois[pid]
        kind, aid = self.active_source
        tier = 3
        if kind == "poi" and aid == pid: tier = 0
        dist = abs(px - poi.tile[0]) + abs(py - poi.tile[1])
        if tier != 0 and self._chebyshev((px,py), poi.tile) <= 1: tier = 1
        return (tier, dist, -poi.rarity)
    
    def _priority_for(self, src: Tuple[str,int]) -> Tuple[int,int]:
        # tier 0: current active source
        if src == self.active_source:
            return (0, 0)
        # tier 1: zones adjacent to player’s zone
        px, py = self.m.player.tile_x, self.m.player.tile_y
        cz = self.m.map.zone_of[px][py]
        adj = set()
        for (nx, ny) in ((px+1,py),(px-1,py),(px,py+1),(px,py-1)):
            if 0 <= nx < self.m.map.w and 0 <= ny < self.m.map.h:
                adj.add(self.m.map.zone_of[nx][ny])
        if src[0] == "zone" and src[1] in adj:
            # distance ~1 for all adjacent zones
            return (1, 1)
        # everything else
        return (2, 99)
        
    def _reprioritize_all(self):
        with self._lock:
            items = list(self._pending.keys())
            self._heap.clear()
            self._pending.clear()
    
        # ensure active first
        rt = self._get_runtime(self.active_source)
        if rt.loop is None and not rt.generating:
            self.request_generate(self.active_source, priority=(0,0), force=True)
    
        # adjacent zones next (no tile scan!)
        if self.active_source[0] == "zone":
            cz = self.active_source[1]
        else:
            # fall back to player's current zone
            cz = self.m.current_zone_id
        for zid in self.m.map.neighbors.get(cz, ()):
            rt = self._get_runtime(("zone", zid))
            if rt.loop is None and not rt.generating:
                self.request_generate(("zone", zid), priority=(1,1), force=False)
    
        # backlog
        for src in items:
            rt = self._get_runtime(src)
            if rt.loop is None and not rt.generating:
                self.request_generate(src, priority=self._priority_for(src), force=False)


    # ---------- cache pruning ----------
    def _prune_cache(self):
        # Count loops
        zone_loops = sum(1 for z in self.m.map.zones.values() if z.loop)
        poi_loops  = sum(1 for p in self.m.map.pois.values() if p.loop)
        total = zone_loops + poi_loops
        if total <= MAX_ACTIVE_LOOPS: return
        # Evict farthest POIs first, then zones
        px, py = self.m.player.tile_x, self.m.player.tile_y
        # Make eviction list of (distance, kind, id)
        cand: List[Tuple[int,str,int]] = []
        for p in self.m.map.pois.values():
            if p.loop and not p.generating:
                d = abs(px-p.tile[0])+abs(py-p.tile[1])
                cand.append((d,"poi",p.pid))
        for zid, z in self.m.map.zones.items():
            if z.loop and not z.generating and zid != self.m.current_zone_id:
                cx, cy = z.centroid
                d = int(abs(px-cx)+abs(py-cy))
                cand.append((d,"zone",zid))
        cand.sort(reverse=True)
        to_remove = total - MAX_ACTIVE_LOOPS
        for _,k,i in cand[:to_remove]:
            if k=="poi":
                self.m.map.pois[i].loop = None
            else:
                self.m.map.zones[i].loop = None

    # ---------- events ----------
    def on_zone_changed(self, old_z, new_z):
        self._maybe_preload_zone(new_z)
        # Force a fresh loop request for the new zone if we don't have one yet
        zr = self.m.map.zones[new_z]
        if zr.loop is None and not zr.generating:
            self.request_zone(new_z, priority=(0,0), force=True)
        self._reprioritize_all()
        self._maybe_switch()


    def on_tile_changed(self, old_t: Tuple[int,int], new_t: Tuple[int,int]):
        self._reprioritize_all()
        self._maybe_switch()

    def _maybe_switch(self):
        next_src = self._pick_audible()
        if next_src != self.active_source:
            self.active_source = next_src
            self._ensure_audible_playing()

    # ---------- preload theme ----------
    def _maybe_preload_zone(self, zid: int):
        z = self.m.map.zones[zid]
        if START_TILE not in z.tiles or z.loop: return
        path = START_THEME_WAV
        if not os.path.isfile(path): return
        try:
            with sf.SoundFile(path) as f:
                duration = len(f) / float(f.samplerate)
        except Exception:
            return
        z.loop = GeneratedLoop(path, duration, z.spec.bpm, z.spec.key_mode, f"Preloaded theme: {z.spec.name}", {"backend":"preloaded"})
        if self.active_source == ("zone", zid):
            self.player.play_loop(z.loop.wav_path, z.loop.duration_sec, cross_ms=180)

    def _maybe_preload_start_zone(self) -> bool:
        k, i = self.active_source
        if k != "zone": return False
        # reuse your START_THEME_WAV logic here
        return self._maybe_preload_zone(i)

    # ---------- recording ----------
    def on_boundary_tick(self):
        if self.record_armed and not self.recorder.is_recording():
            kind, idv = self.active_source
            if kind == "zone":
                z = self.m.map.zones[idv]
                if z.loop:
                    self.recorder.start(z.loop.wav_path, z.loop.duration_sec,
                                        {"zone": z.spec.name, "bpm": z.spec.bpm, "key": z.spec.key_mode, "mood": z.spec.mood})
                    self.record_armed = False
            else:
                p = self.m.map.pois[idv]
                if p.loop:
                    home = self.m.map.zones[p.zone_id].spec.name
                    label = f"{home}__{p.name}"   # no slashes
                    self.recorder.start(
                        p.loop.wav_path,
                        p.loop.duration_sec,
                        {"label": label, "bpm": self.m.map.zones[p.zone_id].spec.bpm,
                         "key": self.m.map.zones[p.zone_id].spec.key_mode, "mood": "funky"}
                    )
                    self.record_armed = False
        elif self.recorder.is_recording() and self.auto_stop_at_end:
            self.recorder.stop()
    
    # Which source is audible right now? (POI within radius 1 beats zone)
    def _current_active_source(self) -> Tuple[str, int]:
        px, py = self.m.player.tile_x, self.m.player.tile_y
        # find nearest POI with Chebyshev distance <= 1
        best = None
        best_d = 999
        for pid, poi in self.m.map.pois.items():
            dx = abs(poi.tile[0] - px)
            dy = abs(poi.tile[1] - py)
            d  = max(dx, dy)
            if d <= 1:
                # prefer NPC over object if tie; otherwise nearest
                rank = (0 if poi.kind == "npc" else 1, d)
                if best is None or rank < best_d:
                    best = ("poi", pid)
                    best_d = rank
        if best: return best
        zid = self.m.map.zone_of[px][py]
        return ("zone", zid)
    
    def _get_runtime(self, src: Tuple[str,int]):
        k, i = src
        if k == "zone":
            return self.m.map.zones[i]
        else:
            return self.m.map.pois[i]
    
    def request_generate(self, src: Tuple[str,int], priority: Tuple[int,int]=(0,0), force: bool=False):
        rt = self._get_runtime(src)
        if not force and (rt.generating or rt.loop is not None or rt.error):
            return
        with self._lock:
            token = next(self._counter)
            self._pending[src] = token
            heapq.heappush(self._heap, (priority, token, src))
            
    def _generate_for(self, src: Tuple[str,int]) -> Optional["GeneratedLoop"]:
        k, i = src
        if k == "zone":
            zr = self.m.map.zones[i]
            return self.provider.generate(zr.spec)
        else:
            poi = self.m.map.pois[i]
            base = self.m.map.zones[poi.zone_id].spec
            bpm  = poi.bpm_hint or random.randint(96, 126)
            bars = poi.bars_hint or base.bars
            zspec = ZoneSpec(
                name=poi.name, bpm=bpm, key_mode=base.key_mode, scene=base.scene,
                mood=(poi.mood_hint or ("energetic" if poi.kind=="npc" else "mysterious")),
                bars=bars, timesig=base.timesig,
                biome=base.biome, species=base.species,
                descriptors=(base.descriptors + (["syncopated","funky"] if poi.kind=="npc" else ["textural"])),
                instruments=(base.instruments + (["rubber synth bass","clavinet","wah guitar"] if poi.kind=="npc" else ["mallets","resonant metal"])),
                tags=list(set(base.tags + (["fast","layered"] if poi.kind=="npc" else ["retro","soft"]))),
            )
            return self.provider.generate(zspec, prepend=tokens_for_poi(poi))
