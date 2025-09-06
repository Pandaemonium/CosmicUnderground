import pygame
from typing import Dict, List, Optional, Tuple
from .config import (
    BOOTLEG_ATTENTION_RATE, SNEAK_ATTENTION_MULTIPLIER, 
    MAX_ATTENTION, ATTENTION_DECAY_RATE, ALIEN_SPECIES,
    EXEC_PREFERENCES
)
from .debug import debug_enter, debug_exit

class GameState:
    def __init__(self):
        self.attention = 0.0
        self.is_sneaking = False
        self.is_recording = False
        self.recording_start_time = 0.0
        self.current_recording_duration = 0.0
        self.money = 100  # Starting money
        self.reputation = {}  # Reputation with each alien species
        self.discovered_descriptors = {}  # Known descriptors for songs
        self.completed_mixes = []  # List of completed song mixes
        self.daughter_rating = 0  # Daughter's rating of current mix (-10 to 10)
        
        # Initialize reputation for each species
        for species in ALIEN_SPECIES:
            self.reputation[species] = 0
        
        # Initialize discovered descriptors
        for exec_name, preferences in EXEC_PREFERENCES.items():
            for pref in preferences:
                self.discovered_descriptors[pref] = False
    
    def update(self, dt: float):
        """Update game state based on time delta"""
        if self.is_recording:
            # Calculate attention gain
            attention_gain = BOOTLEG_ATTENTION_RATE * dt
            if self.is_sneaking:
                attention_gain *= SNEAK_ATTENTION_MULTIPLIER
            
            self.attention = min(MAX_ATTENTION, self.attention + attention_gain)
            self.current_recording_duration += dt
            
            # Check if caught
            if self.attention >= MAX_ATTENTION:
                self.get_caught()
        else:
            # Decay attention when not recording
            self.attention = max(0.0, self.attention - ATTENTION_DECAY_RATE * dt)
    
    def start_recording(self):
        """Start recording a bootleg"""
        debug_enter("start_recording", "game_state.py", current_recording=self.is_recording)
        
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = pygame.time.get_ticks() / 1000.0
            self.current_recording_duration = 0.0
            debug_exit("start_recording", "game_state.py", "Recording started")
        else:
            debug_exit("start_recording", "game_state.py", "Already recording")
    
    def stop_recording(self):
        """Stop recording a bootleg"""
        debug_enter("stop_recording", "game_state.py", current_recording=self.is_recording)
        
        if self.is_recording:
            self.is_recording = False
            duration = self.current_recording_duration
            debug_exit("stop_recording", "game_state.py", f"Recording stopped, duration: {duration}")
            return duration
        
        debug_exit("stop_recording", "game_state.py", "Not recording")
        return 0.0
    
    def toggle_sneaking(self):
        """Toggle sneaking mode"""
        self.is_sneaking = not self.is_sneaking
    
    def get_caught(self):
        """Player got caught bootlegging"""
        self.is_recording = False
        self.attention = 0.0
        self.current_recording_duration = 0.0
        # Lose all bootlegged recordings since last lair visit
        # This will be handled by the inventory system
        return True
    
    def add_money(self, amount: int):
        """Add money to player's account"""
        self.money += amount
    
    def spend_money(self, amount: int) -> bool:
        """Spend money, return True if successful"""
        if self.money >= amount:
            self.money -= amount
            return True
        return False
    
    def discover_descriptor(self, descriptor: str):
        """Mark a descriptor as discovered"""
        self.discovered_descriptors[descriptor] = True
    
    def get_attention_percentage(self) -> float:
        """Get attention as a percentage"""
        return (self.attention / MAX_ATTENTION) * 100
    
    def get_attention_color(self) -> Tuple[int, int, int]:
        """Get color for attention bar (green -> yellow -> red)"""
        percentage = self.get_attention_percentage()
        if percentage < 50:
            # Green to yellow
            r = int(255 * (percentage / 50))
            g = 255
            b = 0
        else:
            # Yellow to red
            r = 255
            g = int(255 * (1 - (percentage - 50) / 50))
            b = 0
        return (r, g, b)
    
    def rate_mix_for_species(self, mix_descriptors: List[str], species: str) -> int:
        """Rate how well a mix matches a species' preferences (0-10)"""
        if species not in ALIEN_SPECIES:
            return 0
        
        species_prefs = ALIEN_SPECIES[species]["preferences"]
        matches = 0
        
        for desc in mix_descriptors:
            if desc.lower() in [pref.lower() for pref in species_prefs]:
                matches += 1
        
        # Calculate rating based on matches
        if matches == 0:
            return 0
        elif matches == 1:
            return 3
        elif matches == 2:
            return 6
        elif matches == 3:
            return 8
        else:
            return 10
    
    def get_daughter_rating(self, mix_descriptors: List[str]) -> int:
        """Get daughter's rating of a mix (-10 to 10)"""
        # Daughter is very picky - she likes unique combinations
        # and hates common/predictable stuff
        rating = 0
        
        # Bonus for unique descriptors
        unique_descriptors = set(mix_descriptors)
        if len(unique_descriptors) >= 3:
            rating += 2
        
        # Bonus for underground/alien descriptors
        underground_words = ["underground", "alien", "cosmic", "ethereal", "otherworldly"]
        for word in underground_words:
            if any(word in desc.lower() for desc in mix_descriptors):
                rating += 1
        
        # Penalty for too many surface-level descriptors
        surface_words = ["pop", "mainstream", "familiar", "earth", "human"]
        surface_count = sum(1 for desc in mix_descriptors 
                          if any(word in desc.lower() for word in surface_words))
        if surface_count > 2:
            rating -= 3
        
        # Random element (daughter is unpredictable)
        import random
        rating += random.randint(-2, 2)
        
        return max(-10, min(10, rating))
    
    def save_to_lair(self):
        """Save current recordings to lair (safe zone)"""
        # Reset attention and recording state
        self.attention = 0.0
        self.is_recording = False
        self.current_recording_duration = 0.0
        return True
