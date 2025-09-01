import pygame
import random
from typing import List
from ..core.config import ZONE_PICKUPS, CRATE_TRACKS, SONG_GROUP_FOLDERS, SUPPORTED_AUDIO_EXTENSIONS, SONG_GROUP_SETTINGS, WIDTH, HEIGHT, BG_COLOR, INFO_COLOR, SPRITE_PATH, TILE_SIZE
from ..core.utils import fmt_time
from ..core.screen import Screen
from ..assets.loader import load_audio_to_array
from ..audio.alien_track import AlienTrack
from ..audio.track import Track
from ..audio.mixer import stop_all_audio
from ..audio.timeline import TimelineState
from ..editor.station_panel import StationPanel
from ..editor.cosmic_daw import CosmicDAW
from .player import Player
from .zones import Zones
import os

class OverworldScreen(Screen):
    def __init__(self, fonts):
        self.FONT, self.SMALL, self.MONO = fonts
        self.player = Player(SPRITE_PATH)
        self.zones = Zones()
        self.active_zone = -1
        self.inventory: List[AlienTrack] = []
        self.timeline = TimelineState()
        self.station = StationPanel(fonts, self.timeline)
        
        # Load background image
        try:
            self.background_image = pygame.image.load("assets/images/325635-light-teal-background.png")
            # Scale the background image to fit the screen dimensions
            self.background_image = pygame.transform.scale(self.background_image, (WIDTH, HEIGHT))
            print("🎨 Background image loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load background image: {e}")
            self.background_image = None
        
        # Crate system
        self.opened_crates = set()  # Track which crates have been opened
        self.crate_prompt_active = False  # Whether to show crate opening prompt
        self.crate_results_active = False  # Whether to show crate results
        self.current_crate_name = None  # Name of crate being opened
        self.crate_tapes_found = []  # Tapes found in current crate
        
        # DAW system
        self.cosmic_daw = None
        self.show_cosmic_daw = False
        
        # Debug: Check what crates are available
        print(f"🎁 CRATE_TRACKS loaded: {list(CRATE_TRACKS.keys())}")
        print(f"🎁 Available crates: {[name for name in self.zones.labels if name.startswith('Crate')]}")
        print(f"🎵 DAW integration ready - press TAB to open Cosmic DAW")

    def handle_event(self, ev):
        # Handle DAW events first if DAW is open
        if self.show_cosmic_daw and self.cosmic_daw and not self.cosmic_daw.should_close:
            # DAW is active - pass all events to it
            self.cosmic_daw.handle_event(ev, self.inventory)
            
            # Check if DAW wants to close
            if self.cosmic_daw.should_close:
                print("🎵 DAW closing, returning to overworld")
                self.cosmic_daw.reset_state()
                self.cosmic_daw = None
                self.show_cosmic_daw = False
                return
            
            # DAW is active, don't process overworld events
            return
        
        # Handle crate prompts and results first
        if self.crate_prompt_active:
            self._handle_crate_prompt_event(ev)
            return
        elif self.crate_results_active:
            self._handle_crate_results_event(ev)
            return
        
        if ev.type == pygame.KEYDOWN:
            # Handle Tab key to open DAW
            if ev.key == pygame.K_TAB:
                print("🎵 TAB key pressed - Opening Cosmic DAW...")
                if not self.show_cosmic_daw:
                    self.cosmic_daw = CosmicDAW((self.FONT, self.SMALL, self.MONO), self)
                    self.show_cosmic_daw = True
                return
            
            # If we're standing in any pickup zone, allow 1..4 to grab local files for that zone
            if self.active_zone >= 0:
                zone_name = self.zones.labels[self.active_zone]
                
                # Check if this is a crate that hasn't been opened yet
                if zone_name.startswith("Crate"):
                    if zone_name not in self.opened_crates:
                        if ev.key == pygame.K_SPACE:  # Space to open crate
                            print(f"🎁 SPACE pressed on crate {zone_name} - starting crate opening!")
                            self._start_crate_opening(zone_name)
                            return
                        else:
                            print(f"🎁 On crate {zone_name} - press SPACE to open")
                    else:
                        print(f"🎁 Crate {zone_name} already opened")
                
                # Handle regular zone pickups
                file_list = ZONE_PICKUPS.get(zone_name)
                if file_list:
                    if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        idx = {pygame.K_1:0, pygame.K_2:1, pygame.K_3:2, pygame.K_4:3}[ev.key]
                        path = file_list[idx] if idx < len(file_list) else None
                        if path:
                            loaded = load_audio_to_array(path)
                            if loaded:
                                name, arr = loaded
                                self.inventory.append(AlienTrack(name, arr, source_zone=zone_name))
                                print(f"Picked up from {zone_name}:", name)
                            else:
                                print("Failed to load:", path)
            elif self.active_zone == 1:  # Listening Station
                self.station.handle_key(ev, self.inventory, lambda: stop_all_audio(self.inventory, self.timeline.active_channels))
        elif ev.type == pygame.MOUSEBUTTONDOWN and self.active_zone == 1:
            if ev.button in (1,2):
                self.station.handle_mouse_down(ev.pos, ev.button, self.inventory)
        elif ev.type == pygame.MOUSEBUTTONUP and self.active_zone == 1:
            self.station.handle_mouse_up(ev.button)
        elif ev.type == pygame.MOUSEMOTION and self.active_zone == 1:
            self.station.handle_mouse_move(ev.pos, self.inventory)
        elif ev.type == pygame.MOUSEWHEEL and self.active_zone == 1:
            self.station.handle_wheel(ev, self.inventory)

    def update(self, dt):
        keys = pygame.key.get_pressed()
        self.player.update(dt, keys, WIDTH, HEIGHT)
        inside = self.zones.which(self.player.pos)
        if inside != self.active_zone:
            if self.active_zone == 1 and inside != 1:
                # leaving station: stop audio + timeline
                stop_all_audio(self.inventory, self.timeline.active_channels)
                self.timeline.stop()
            
            # Debug zone changes
            old_zone = self.active_zone
            self.active_zone = inside
            
            if inside >= 0:
                new_zone_name = self.zones.labels[inside]
                
                # Check if entering a crate for the first time
                if new_zone_name.startswith("Crate") and new_zone_name not in self.opened_crates:
                    print(f"🎁 First time on crate {new_zone_name} - auto-triggering prompt!")
                    self._start_crate_opening(new_zone_name)
        # update audition playhead
        if self.active_zone == 1:
            self.station.audition_update_cursor()
            self.timeline.update()
        
        # Update DAW if active
        if self.show_cosmic_daw and self.cosmic_daw:
            self.cosmic_daw.update(dt)

    def draw(self, screen):
        if self.background_image:
            screen.blit(self.background_image, (0, 0))
        else:
            screen.fill(BG_COLOR)
        # world
        self.zones.draw(screen, self.FONT)
        self.player.draw(screen)
        # HUD
        y = 8
        screen.blit(self.SMALL.render(f"Inventory: {len(self.inventory)}", True, INFO_COLOR), (12, y)); y += 22
        tips = [
            "Move: WASD/Arrows",
            "TAB: Open Cosmic DAW",
            "Mint Lagoon: 1..4 pick up local files (paths in config.py).",
            "Crates: SPACE to open when standing on them.",
            "Listening Station: select track, set selection, Space=audition, A=Add@Cursor.",
            "Timeline: click lane to target; wheel=zoom, Shift+wheel=pan, ▶/⏹ play/stop."
        ]
        for t in tips:
            screen.blit(self.SMALL.render(t, True, INFO_COLOR), (12, y)); y += 18
        # station UI overlay
        if self.active_zone == 1:
            self.station.render(screen, self.inventory, WIDTH, HEIGHT)
        
        # Crate UI overlays
        if self.crate_prompt_active:
            self._render_crate_prompt(screen)
        elif self.crate_results_active:
            self._render_crate_results(screen)
        
        # DAW overlay
        if self.show_cosmic_daw and self.cosmic_daw:
            self.cosmic_daw.render(screen, self.inventory)
    
    def _start_crate_opening(self, crate_name):
        """Start the crate opening process"""
        print(f"🎁 _start_crate_opening called with: {crate_name}")
        self.current_crate_name = crate_name
        self.crate_prompt_active = True
        print(f"🎁 Crate prompt active: {self.crate_prompt_active}")
        print(f"🎁 Found {crate_name}! Press SPACE to open it.")
    
    def _handle_crate_prompt_event(self, ev):
        """Handle events while crate opening prompt is active"""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                # Open the crate
                self._open_crate()
            elif ev.key == pygame.K_ESCAPE:
                # Cancel crate opening
                self.crate_prompt_active = False
                self.current_crate_name = None
    
    def _handle_crate_results_event(self, ev):
        """Handle events while crate results are shown"""
        if ev.type == pygame.KEYDOWN:
            if ev.key in [pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE]:
                # Close results and add tapes to inventory
                self._finish_crate_opening()
    
    def _expand_song_groups(self, crate_tracks):
        """
        Expand song group folders into individual audio files.
        Returns a list of all individual audio file paths.
        """
        expanded_tracks = []
        
        for track_path in crate_tracks:
            if os.path.isdir(track_path):
                # This is a song group folder - expand it
                print(f"🎵 Expanding song group: {track_path}")
                try:
                    # Get all audio files from the folder
                    audio_files = []
                    for filename in os.listdir(track_path):
                        file_path = os.path.join(track_path, filename)
                        if os.path.isfile(file_path):
                            # Check if it's a supported audio file
                            file_ext = os.path.splitext(filename)[1].lower()
                            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                                audio_files.append(file_path)
                    
                    print(f"🎵 Found {len(audio_files)} audio files in song group")
                    expanded_tracks.extend(audio_files)
                    
                except Exception as e:
                    print(f"⚠️ Error expanding song group {track_path}: {e}")
                    # Fallback: add the folder path as-is
                    expanded_tracks.append(track_path)
            else:
                # This is an individual file - add it directly
                expanded_tracks.append(track_path)
        
        return expanded_tracks

    def _open_crate(self):
        """Open the crate and determine what tapes are found"""
        crate_name = self.current_crate_name
        
        # Get random tapes from CRATE_TRACKS
        if crate_name in CRATE_TRACKS:
            available_tracks = CRATE_TRACKS[crate_name]
            
            # Expand song group folders into individual files
            expanded_tracks = self._expand_song_groups(available_tracks)
            print(f"🎵 Crate {crate_name} has {len(expanded_tracks)} total tracks after expansion")
            
            # Determine which tracks to give based on song group settings
            tracks_to_give = []
            
            if SONG_GROUP_SETTINGS["GIVE_ALL_TRACKS_FROM_GROUPS"]:
                # Check if any song groups were found
                song_group_tracks = []
                individual_tracks = []
                
                for track_path in expanded_tracks:
                    # Check if this track came from a song group
                    is_from_group = False
                    for song_group in SONG_GROUP_FOLDERS:
                        if song_group in track_path:
                            song_group_tracks.append(track_path)
                            is_from_group = True
                            break
                    
                    if not is_from_group:
                        individual_tracks.append(track_path)
                
                # If we found song group tracks, give ALL of them
                if song_group_tracks:
                    tracks_to_give.extend(song_group_tracks)
                    print(f"🎵 Giving ALL {len(song_group_tracks)} tracks from song groups")
                
                # Add some individual tracks if we have room
                if individual_tracks:
                    max_individual = SONG_GROUP_SETTINGS["MAX_INDIVIDUAL_TRACKS"]
                    num_individual = min(max_individual, len(individual_tracks))
                    selected_individual = random.sample(individual_tracks, num_individual)
                    tracks_to_give.extend(selected_individual)
                    print(f"🎵 Adding {num_individual} individual tracks")
                
                # Ensure we meet minimum track requirement
                if len(tracks_to_give) < SONG_GROUP_SETTINGS["MIN_TRACKS_PER_CRATE"]:
                    remaining_tracks = [t for t in expanded_tracks if t not in tracks_to_give]
                    if remaining_tracks:
                        needed = SONG_GROUP_SETTINGS["MIN_TRACKS_PER_CRATE"] - len(tracks_to_give)
                        additional = random.sample(remaining_tracks, min(needed, len(remaining_tracks)))
                        tracks_to_give.extend(additional)
                        print(f"🎵 Adding {len(additional)} additional tracks to meet minimum")
            else:
                # Original behavior: random selection
                num_tracks = random.randint(
                    SONG_GROUP_SETTINGS["MIN_TRACKS_PER_CRATE"], 
                    min(SONG_GROUP_SETTINGS["MAX_INDIVIDUAL_TRACKS"], len(expanded_tracks))
                )
                tracks_to_give = random.sample(expanded_tracks, num_tracks)
                print(f"🎵 Randomly selected {num_tracks} tracks")
            
            # Load the selected tracks
            self.crate_tapes_found = []
            for track_path in tracks_to_give:
                loaded = load_audio_to_array(track_path)
                if loaded:
                    name, arr = loaded
                    track = AlienTrack(name, arr, source_zone=crate_name)
                    self.crate_tapes_found.append(track)
                    print(f"🎵 Found in {crate_name}: {name}")
                else:
                    print(f"⚠️ Failed to load track from {crate_name}: {track_path}")
        else:
            # Fallback: create some placeholder tracks
            self.crate_tapes_found = [
                AlienTrack("Mystery Tape 1", None, source_zone=crate_name),
                AlienTrack("Mystery Tape 2", None, source_zone=crate_name),
                AlienTrack("Mystery Tape 3", None, source_zone=crate_name)
            ]
            print(f"🎵 Found mystery tapes in {crate_name}")
        
        # Mark crate as opened
        self.opened_crates.add(crate_name)
        
        # Switch to results view
        self.crate_prompt_active = False
        self.crate_results_active = True
    
    def _finish_crate_opening(self):
        """Finish crate opening and add tapes to inventory"""
        # Add all found tapes to inventory
        for track in self.crate_tapes_found:
            self.inventory.append(track)
        
        print(f"🎁 Added {len(self.crate_tapes_found)} tapes to inventory from {self.current_crate_name}")
        
        # Reset crate state
        self.crate_results_active = False
        self.current_crate_name = None
        self.crate_tapes_found = []
    
    def _render_crate_prompt(self, screen):
        """Render the crate opening prompt"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Prompt box
        box_width, box_height = 400, 200
        box_x = (WIDTH - box_width) // 2
        box_y = (HEIGHT - box_height) // 2
        
        # Box background
        pygame.draw.rect(screen, (40, 45, 60), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (100, 150, 255), (box_x, box_y, box_width, box_height), 3)
        
        # Title
        title = self.FONT.render(f"🎁 {self.current_crate_name} Found!", True, (255, 255, 255))
        title_rect = title.get_rect(center=(WIDTH // 2, box_y + 40))
        screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "You found a mysterious crate!",
            "Press SPACE to open it and see what's inside.",
            "Press ESC to leave it for later."
        ]
        
        y_offset = box_y + 80
        for instruction in instructions:
            text = self.SMALL.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect(center=(WIDTH // 2, y_offset))
            screen.blit(text, text_rect)
            y_offset += 25
    
    def _render_crate_results(self, screen):
        """Render the crate opening results"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Results box - make it larger to accommodate song group info
        box_width, box_height = 600, 350
        box_x = (WIDTH - box_width) // 2
        box_y = (HEIGHT - box_height) // 2
        
        # Box background
        pygame.draw.rect(screen, (40, 45, 60), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (100, 150, 255), (box_x, box_y, box_width, box_height), 3)
        
        # Title
        title = self.FONT.render(f"🎵 Tapes Found in {self.current_crate_name}!", True, (255, 255, 255))
        title_rect = title.get_rect(center=(WIDTH // 2, box_y + 40))
        screen.blit(title, title_rect)
        
        # List found tapes
        y_offset = box_y + 80
        song_groups_found = set()
        
        for i, track in enumerate(self.crate_tapes_found, 1):
            # Check if this track came from a song group
            track_name = track.name
            track_source = getattr(track, 'source_zone', '')
            
            # Try to determine if it came from a song group by checking the track name
            # (since we don't store the full path in the track object)
            is_from_group = False
            for song_group in SONG_GROUP_FOLDERS:
                group_name = os.path.basename(song_group)
                if group_name.lower() in track_name.lower():
                    is_from_group = True
                    song_groups_found.add(group_name)
                    break
            
            if is_from_group:
                track_name += " 🎼"  # Add music note emoji for song group tracks
            
            text = self.SMALL.render(f"{i}. {track_name}", True, (200, 200, 200))
            text_rect = text.get_rect(center=(WIDTH // 2, y_offset))
            screen.blit(text, text_rect)
            y_offset += 25
        
        # Show song group info if any were found
        if song_groups_found:
            y_offset += 10
            group_text = self.FONT.render("🎼 Complete Song Groups Found!", True, (100, 200, 255))
            group_rect = group_text.get_rect(center=(WIDTH // 2, y_offset))
            screen.blit(group_text, group_rect)
            y_offset += 25
            
            for group_name in sorted(song_groups_found):
                group_info = self.SMALL.render(f"• {group_name} - All tracks included!", True, (150, 200, 255))
                group_info_rect = group_info.get_rect(center=(WIDTH // 2, y_offset))
                screen.blit(group_info, group_info_rect)
                y_offset += 20
        
        # Instructions
        instructions = [
            "Press SPACE, ENTER, or ESC to add tapes to inventory."
        ]
        
        y_offset = box_y + box_height - 60
        for instruction in instructions:
            text = self.SMALL.render(instruction, True, (150, 150, 150))
            text_rect = text.get_rect(center=(WIDTH // 2, y_offset))
            screen.blit(text, text_rect)
            y_offset += 20
