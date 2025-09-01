from typing import Dict, List, Optional, Tuple
import pygame
import numpy as np
import random
from .track import Track
from ..core.config import SR, MIN_CLIP_SEC, ALIEN_SPECIES
from ..core.debug import debug_enter, debug_exit

class AlienTrack(Track):
    def __init__(self, name: str, array: np.ndarray, source_zone: str = "", is_bootleg: bool = False):
        debug_enter("AlienTrack.__init__", "alien_track.py", name=name, source_zone=source_zone, is_bootleg=is_bootleg)
        
        super().__init__(name, array)
        self.source_zone = source_zone
        self.is_bootleg = is_bootleg
        self.descriptors = self._generate_descriptors()
        self.species_ratings = self._calculate_species_ratings()
        self.recording_quality = self._calculate_recording_quality()
        self.value = self._calculate_value()
        
        debug_exit("AlienTrack.__init__", "alien_track.py", f"Track created with {len(self.descriptors)} descriptors, value: ${self.value}")
        
    def _generate_descriptors(self) -> List[str]:
        """Generate descriptors for the track based on source and type"""
        descriptors = []
        
        # Base descriptors based on source zone
        zone_descriptors = {
            "Mint Lagoon": ["surface", "familiar", "accessible", "mainstream"],
            "Underground Caves": ["underground", "organic", "natural", "cave"],
            "Deep Underground": ["deep", "industrial", "mechanical", "underground"],
            "High Altitude": ["airy", "light", "floating", "high"],
            "Forest Grove": ["organic", "natural", "earthy", "forest"],
            "Polar Ice Cap": ["cold", "crystalline", "frozen", "polar"],
            "Your Lair": ["safe", "familiar", "home", "comfortable"]
        }
        
        if self.source_zone in zone_descriptors:
            descriptors.extend(zone_descriptors[self.source_zone])
        
        # Add bootleg-specific descriptors
        if self.is_bootleg:
            bootleg_descriptors = ["bootleg", "illegal", "raw", "authentic", "underground"]
            descriptors.extend(random.sample(bootleg_descriptors, 2))
        
        # Add random musical descriptors
        musical_descriptors = [
            "rhythmic", "melodic", "harmonic", "percussive", "ambient",
            "electronic", "organic", "mechanical", "ethereal", "grounded",
            "fast", "slow", "intense", "relaxing", "energetic", "calm"
        ]
        descriptors.extend(random.sample(musical_descriptors, 3))
        
        # Add alien-specific descriptors
        alien_descriptors = [
            "cosmic", "otherworldly", "alien", "extraterrestrial", "galactic",
            "dimensional", "quantum", "stellar", "nebular", "void"
        ]
        descriptors.extend(random.sample(alien_descriptors, 2))
        
        return list(set(descriptors))  # Remove duplicates
    
    def _calculate_species_ratings(self) -> Dict[str, int]:
        """Calculate how well this track matches each species' preferences"""
        ratings = {}
        
        for species_name, species_data in ALIEN_SPECIES.items():
            species_prefs = species_data["preferences"]
            matches = 0
            
            for desc in self.descriptors:
                if desc.lower() in [pref.lower() for pref in species_prefs]:
                    matches += 1
            
            # Calculate rating (0-10)
            if matches == 0:
                rating = 0
            elif matches == 1:
                rating = 2
            elif matches == 2:
                rating = 4
            elif matches == 3:
                rating = 6
            elif matches == 4:
                rating = 8
            else:
                rating = 10
            
            ratings[species_name] = rating
        
        return ratings
    
    def _calculate_recording_quality(self) -> float:
        """Calculate recording quality (0.0 to 1.0)"""
        if not self.is_bootleg:
            return 1.0  # Professional recordings are perfect
        
        # Bootleg quality depends on various factors
        quality = 0.5  # Base bootleg quality
        
        # Longer recordings might be lower quality
        if self.duration > 30:
            quality -= 0.1
        
        # Underground recordings might be better (more authentic)
        if "underground" in self.descriptors:
            quality += 0.2
        
        # Random variation
        quality += random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, quality))
    
    def _calculate_value(self) -> int:
        """Calculate the monetary value of this track"""
        base_value = 10
        
        # Bootlegs are worth more
        if self.is_bootleg:
            base_value *= 2
        
        # Underground tracks are worth more
        if "underground" in self.descriptors:
            base_value *= 1.5
        
        # Rarer descriptors increase value
        rare_descriptors = ["cosmic", "quantum", "dimensional", "void", "nebular"]
        for desc in rare_descriptors:
            if desc in self.descriptors:
                base_value += 5
        
        # Quality affects value
        base_value = int(base_value * self.recording_quality)
        
        return max(5, base_value)
    
    def get_best_species_match(self) -> Tuple[str, int]:
        """Get the species that would like this track the most"""
        best_species = None
        best_rating = -1
        
        for species, rating in self.species_ratings.items():
            if rating > best_rating:
                best_rating = rating
                best_species = species
        
        return best_species, best_rating
    
    def get_descriptor_summary(self) -> str:
        """Get a human-readable summary of the track's descriptors"""
        if not self.descriptors:
            return "No descriptors available"
        
        # Group descriptors by category
        categories = {
            "Location": ["surface", "underground", "deep", "high", "polar", "forest"],
            "Style": ["rhythmic", "melodic", "electronic", "organic", "mechanical"],
            "Mood": ["energetic", "calm", "intense", "relaxing", "fast", "slow"],
            "Origin": ["cosmic", "alien", "otherworldly", "galactic", "dimensional"]
        }
        
        summary_parts = []
        for category, category_descs in categories.items():
            matching = [desc for desc in self.descriptors if desc in category_descs]
            if matching:
                summary_parts.append(f"{category}: {', '.join(matching[:2])}")
        
        return " | ".join(summary_parts)
    
    def can_combine_with(self, other_track: 'AlienTrack') -> bool:
        """Check if this track can be combined with another track"""
        # Tracks can be combined if they have some compatible descriptors
        # or if one is surface and one is underground
        surface_words = ["surface", "familiar", "mainstream", "accessible"]
        underground_words = ["underground", "deep", "alien", "cosmic"]
        
        has_surface = any(word in self.descriptors for word in surface_words)
        has_underground = any(word in self.descriptors for word in underground_words)
        
        other_has_surface = any(word in other_track.descriptors for word in surface_words)
        other_has_underground = any(word in other_track.descriptors for word in underground_words)
        
        # Good combination: surface + underground
        if (has_surface and other_has_underground) or (has_underground and other_has_surface):
            return True
        
        # Good combination: similar style descriptors
        style_words = ["rhythmic", "melodic", "electronic", "organic"]
        self_styles = [desc for desc in self.descriptors if desc in style_words]
        other_styles = [desc for desc in other_track.descriptors if desc in style_words]
        
        if self_styles and other_styles and any(style in other_styles for style in self_styles):
            return True
        
        return False
    
    def create_mix_with(self, other_track: 'AlienTrack') -> 'AlienTrack':
        """Create a new mixed track combining this track with another"""
        # This is a simplified mix - in a real implementation you'd do audio mixing
        mix_name = f"{self.name} + {other_track.name} Mix"
        
        # Combine arrays (simple concatenation for now)
        # In reality, you'd want proper audio mixing
        combined_array = np.concatenate([self.array, other_track.array])
        
        # Create mixed track
        mixed_track = AlienTrack(mix_name, combined_array, "mixed", True)
        
        # Combine descriptors
        all_descriptors = list(set(self.descriptors + other_track.descriptors))
        mixed_track.descriptors = all_descriptors
        
        # Recalculate ratings and value
        mixed_track.species_ratings = mixed_track._calculate_species_ratings()
        mixed_track.value = mixed_track._calculate_value()
        
        return mixed_track
