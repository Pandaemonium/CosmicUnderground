from typing import List, Tuple, Dict
from cosmic_underground.core.models import ZoneRuntime, POI, ZoneSpec
from cosmic_underground.core.config import *
import random


# ======= Region-growing map + ProcGen =======
class RegionMap:
    """Builds contiguous zone blobs and places POIs, including Boss Skuggs in start zone."""
    ALIENS = {
        "Chillaxians": dict(biomes=["Polar Ice","Crystal Canyons"], bpm=(84,102), mood=["calm","triumphant"],
                            tags=["slow","retro","soft"], inst=["warm pads","ice chimes","soft bass"]),
        "Glorpals":    dict(biomes=["Slime Caves","Groove Pits"],   bpm=(96,122), mood=["playful","melancholy"],
                            tags=["wet","slurpy","layered"], inst=["gel bells","drip fx","squelch synth"]),
        "Bzaris":      dict(biomes=["Centerspire","Crystal Canyons"], bpm=(120,140), mood=["energetic","angry"],
                            tags=["fast","glitchy","futuristic"], inst=["bitcrush blips","granular stutters","noisy hats"]),
        "Shagdeliacs": dict(biomes=["Groove Pits","Centerspire"],   bpm=(96,120), mood=["funky","playful"],
                            tags=["brassy","jazzy","retro"], inst=["wah guitar","clavinet","horn section"]),
        "Rockheads":   dict(biomes=["Groove Pits","Slime Caves"],   bpm=(100,126), mood=["brooding","triumphant"],
                            tags=["mechanical","metallic","loud"], inst=["metal gongs","anvil hits","deep toms"]),
    }
    BIOMES = ["Crystal Canyons","Slime Caves","Centerspire","Groove Pits","Polar Ice","Mag-Lev Yards","Electro Marsh"]
    MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]
    TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
    
    ALIEN_FIRST = ["Skuggs", "Jax", "Zor", "Blee", "Kru", "Vex", "Talla", "Moxo", "Rz-7", "Floop"]
    ALIEN_LAST  = ["of Bazaria", "the Fuzzy", "from Groovepit", "Chrome-Snout", "Mag-Skipper", "Buzzwing", "Chilldraft"]
    OBJECT_NAMES = {
        "amp": ["Chromeblaster", "Groove Reactor", "Ion Stack", "Neon Cab", "Riff Turbine"],
        "boombox": ["Slimebox", "Funk Beacon", "Pulse Cube", "Laser Luggable"],
        "gong": ["Shard Gong", "Cavern Gong", "Mag Gong", "Phase Gong"],
        "terminal": ["Beat Kiosk", "Loop Terminal", "Wave Vender", "Rhythm Post"],
    }

    def __init__(self, w: int, h: int, seed: int):
            self.w, self.h = w, h
            self.seed = seed
            self.rng = random.Random(seed)
    
            self.zone_of: List[List[int]] = [[-1]*h for _ in range(w)]
            self.zones: Dict[int, ZoneRuntime] = {}
            self.pois: Dict[int, POI] = {}
            self.pois_at: Dict[Tuple[int,int], int] = {}
    
            # ✅ make this BEFORE _procgen_zone_specs uses it
            self.zone_color: Dict[int, Tuple[int,int,int]] = {}
    
            self._build_regions()
            self._procgen_zone_specs()
            self._place_pois()

    # --- Region growing ---
    def _build_regions(self):
        total_tiles = self.w * self.h
        target_zones = max(1, total_tiles // AVG_ZONE)
        # Build zone budgets 10..40 until we cover the map
        budgets = []
        s = 0
        while s < total_tiles:
            k = self.rng.randint(ZONE_MIN, ZONE_MAX)
            budgets.append(k); s += k
        # Seeds: choose distinct unassigned tiles
        unassigned = {(x, y) for x in range(self.w) for y in range(self.h)}
        seeds = []
        for _ in range(len(budgets)):
            if not unassigned: break
            t = self.rng.choice(tuple(unassigned))
            seeds.append(t)
            unassigned.remove(t)

        # Initialize frontier per zone id
        zone_id = 0
        frontiers: Dict[int, List[Tuple[int,int]]] = {}
        want: Dict[int, int] = {}
        for s_tile, budget in zip(seeds, budgets):
            x, y = s_tile
            self.zone_of[x][y] = zone_id
            want[zone_id] = max(ZONE_MIN, min(ZONE_MAX, budget))
            frontiers[zone_id] = [s_tile]
            zone_id += 1

        # Multi-source random BFS growth
        active = set(frontiers.keys())
        while active:
            zid = self.rng.choice(tuple(active))
            if want[zid] <= 0:
                active.remove(zid); continue
            if not frontiers[zid]:
                active.remove(zid); continue

            fx, fy = frontiers[zid].pop(0)
            # neighbors (4-neighborhood; diagonals allowed to "touch only at corners" naturally)
            for nx, ny in ((fx+1,fy),(fx-1,fy),(fx,fy+1),(fx,fy-1)):
                if 0 <= nx < self.w and 0 <= ny < self.h and self.zone_of[nx][ny] == -1:
                    self.zone_of[nx][ny] = zid
                    frontiers[zid].append((nx, ny))
                    want[zid] -= 1
                    if want[zid] <= 0: break
            if want[zid] <= 0:
                active.remove(zid)

        # Assign any leftovers to nearest assigned neighbor
        for x in range(self.w):
            for y in range(self.h):
                if self.zone_of[x][y] != -1: continue
                # pick random neighboring assigned
                cand = []
                for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                    if 0 <= nx < self.w and 0 <= ny < self.h and self.zone_of[nx][ny] != -1:
                        cand.append(self.zone_of[nx][ny])
                self.zone_of[x][y] = self.rng.choice(cand) if cand else 0

        # Build zone tile lists
        tiles_by_zone: Dict[int, List[Tuple[int,int]]] = {}
        for x in range(self.w):
            for y in range(self.h):
                zid = self.zone_of[x][y]
                tiles_by_zone.setdefault(zid, []).append((x,y))
        # Make ZoneRuntime shells (specs filled next)
        for zid, tl in tiles_by_zone.items():
            cx = sum(t[0] for t in tl) / len(tl)
            cy = sum(t[1] for t in tl) / len(tl)
            self.zones[zid] = ZoneRuntime(zid, None, tl, (cx, cy), None, False, None)
        
        # Stable label anchor per zone (top-left tile in world coords)
        self.zone_anchor: Dict[int, Tuple[int,int]] = {}
        for zid, tl in tiles_by_zone.items():
            self.zone_anchor[zid] = min(tl)  # lexicographic = top-left
        
        self.neighbors: Dict[int, set[int]] = {zid:set() for zid in self.zones}
        for x in range(self.w):
            for y in range(self.h):
                a = self.zone_of[x][y]
                for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        b = self.zone_of[nx][ny]
                        if a != b:
                            self.neighbors[a].add(b)
                            self.neighbors[b].add(a)

    def _procgen_zone_specs(self):
        for zid, zr in self.zones.items():
            rng = random.Random((self.seed << 1) ^ (zid * 2654435761))
            biome = rng.choice(self.BIOMES)
            species = rng.choice(list(self.ALIENS.keys()))
            # prefer matching biome
            for _ in range(2):
                p = rng.choice(list(self.ALIENS.keys()))
                if biome in self.ALIENS[p]["biomes"]:
                    species = p; break
            a = self.ALIENS[species]
            bpm = rng.randrange(a["bpm"][0], a["bpm"][1]+1, 2)
            mood = rng.choice(a["mood"])
            tags = list(set(a["tags"]))
            instruments = list(set(a["inst"]))
            # name
            if START_TILE in zr.tiles:
                name = START_ZONE_NAME
            else:
                A = ["Neon","Crystal","Slime","Chrome","Velvet","Magnetic","Shatter","Rusty","Electric","Gleaming","Polar","Echo"]
                L = ["Bazaar","Canyons","Caves","Centerspire","Grove","Pits","Foundry","Sprawl","Galleries","Arcade","Causeway","Yards"]
                T = ["Funk","Pulse","Flux","Jitter","Parade","Breaks","Mallets","Reverb","Drift","Chromatics","Riff"]
                base = f"{rng.choice(A)} {rng.choice(L)}"
                if rng.random() < 0.4: base = f"{base} {rng.choice(T)}"
                name = base
            key_mode = f"{rng.choice(self.TONICS)} {rng.choice(self.MODES)}"
            scene = rng.choice([
                "glittering arps and clacky percussion",
                "dubby chords and foghorn pads",
                "glassy mallets and airy choirs",
                "breakbeat kit and gnarly clavs",
                "hand percussion and kalimba",
                "trip-hop haze with vinyl crackle",
                "UKG shuffle with vocal chops",
                "parade brass and wah guitar",
                "holo-arcade bleeps and neon hum",
            ])
            zr.spec = ZoneSpec(
                name=name, bpm=bpm, key_mode=key_mode, scene=scene, mood=mood,
                biome=biome, species=species, descriptors=[], instruments=instruments, tags=tags
            )
            rr = random.Random(zid*1337)
            self.zone_color[zid] = (40 + rr.randint(0,40), 42 + rr.randint(0,30), 70 + rr.randint(0,40))

    # --- POIs ---
    def _place_pois(self):
        next_id = 1
        rng = self.rng
    
        def interior_tiles(zr):
            tl = zr.tiles
            if len(tl) < 4: return tl
            sx = sum(x for x,_ in tl)/len(tl); sy = sum(y for _,y in tl)/len(tl)
            return sorted(tl, key=lambda t: abs(t[0]-sx)+abs(t[1]-sy))
    
        def make_npc_name(rng, species: str) -> str:
            if species == "Bzaris":
                return f"{rng.choice(['Bz-','Zz-','Q-'])}{rng.choice(['Skug','Zap','Kli','Vrr'])}{rng.randrange(10,99)}"
            if species == "Glorpals":
                return f"{rng.choice(['Glo','Slu','Dro'])}{rng.choice(['rpo','rma','opa'])}{rng.choice(['x','z'])}"
            return f"{rng.choice(self.ALIEN_FIRST)} {rng.choice(self.ALIEN_LAST)}"
    
        # Boss in start zone
        for zid, zr in self.zones.items():
            if (self.START_X, self.START_Y) if hasattr(self, "START_X") else None:  # ignore if not set
                pass
            if (self.w//2, self.h//2) in zr.tiles or True:
                home = interior_tiles(zr)[0]
                self.pois[next_id] = POI(next_id, "npc", "Boss Skuggs", "boss", home, zid, rarity=10)
                self.pois_at[home] = next_id
                next_id += 1
                break
    
        for zid, zr in self.zones.items():
            # counts
            if self.START_TILE in zr.tiles if hasattr(self, "START_TILE") else False:
                npc_n = rng.randint(max(0, POIS_NPC_RANGE[0]-1), max(1, POIS_NPC_RANGE[1]))
                obj_n = rng.randint(*POIS_OBJ_RANGE)
            else:
                npc_n = rng.randint(*POIS_NPC_RANGE)
                obj_n = rng.randint(*POIS_OBJ_RANGE)
    
            candidates = [t for t in interior_tiles(zr) if t not in self.pois_at]
            rng.shuffle(candidates)
    
            # NPCs
            for _ in range(npc_n):
                if not candidates: break
                tile = candidates.pop(0)
                name = make_npc_name(rng, zr.spec.species if zr.spec else "Unknown")
                self.pois[next_id] = POI(next_id, "npc", name, "performer", tile, zid, rarity=rng.randint(0,3))
                self.pois_at[tile] = next_id
                next_id += 1
    
            # Objects
            for _ in range(obj_n):
                if not candidates: break
                tile = candidates.pop(0)
                name = rng.choice(["Crystal Resonator","Slime Drum","Mag Lev Bell","Arcade Cabinet"])
                self.pois[next_id] = POI(next_id, "object", name, "resonator", tile, zid, rarity=rng.randint(0,2))
                self.pois_at[tile] = next_id
                next_id += 1
    
        # Quest-giver beacon in a neighbor zone of the start
        start_zid = self.zone_of[START_TILE[0]][START_TILE[1]]
        neighs = list(self.neighbors.get(start_zid, []))
        rng.shuffle(neighs)
        for qz in neighs:
            zr = self.zones[qz]
            tiles = [t for t in interior_tiles(zr) if t not in self.pois_at] or zr.tiles[:]
            if not tiles: continue
            tile = tiles[0]
            beacon = POI(next_id, "object", "Beacon of Names", "quest_giver", tile, qz, rarity=99, kind_key="beacon")
            self.pois[next_id] = beacon
            self.pois_at[tile] = next_id
            self.quest_giver_pid = next_id
            next_id += 1
            break