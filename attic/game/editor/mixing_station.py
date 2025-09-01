import pygame
from typing import List, Tuple, Optional
from ..core.config import (
    WIDTH, HEIGHT, TEXT_COLOR, INFO_COLOR, UI_OUTLINE, HILITE, ERR,
    ALIEN_SPECIES, EXEC_PREFERENCES
)
from ..core.game_state import GameState
from ..audio.alien_track import AlienTrack

class MixingStation:
    def __init__(self, fonts, game_state: GameState):
        self.FONT, self.SMALL, self.MONO = fonts
        self.game_state = game_state
        self.selected_tracks: List[AlienTrack] = []
        self.current_mix: Optional[AlienTrack] = None
        self.show_sales_panel = False
        self.show_mix_preview = False
        
        # Panel dimensions
        self.panel_width = 400
        self.panel_height = 500
        self.panel_x = (WIDTH - self.panel_width) // 2
        self.panel_y = (HEIGHT - self.panel_height) // 2
        
        # Button dimensions
        self.button_height = 35
        self.button_margin = 8
        
    def handle_event(self, ev, inventory: List[AlienTrack]):
        """Handle mouse and keyboard events"""
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:  # Left click
                self._handle_mouse_click(ev.pos, inventory)
        elif ev.type == pygame.KEYDOWN:
            self._handle_key_press(ev, inventory)
    
    def _handle_mouse_click(self, pos: Tuple[int, int], inventory: List[AlienTrack]):
        """Handle mouse clicks on the mixing station"""
        x, y = pos
        
        # Check if click is within panel bounds
        if not (self.panel_x <= x <= self.panel_x + self.panel_width and
                self.panel_y <= y <= self.panel_y + self.panel_height):
            return
        
        # Calculate relative position within panel
        rel_x = x - self.panel_x
        rel_y = y - self.panel_y
        
        # Track selection area (left side)
        if rel_x < 200:
            track_index = (rel_y - 50) // 40
            if 0 <= track_index < len(inventory):
                track = inventory[track_index]
                if track in self.selected_tracks:
                    self.selected_tracks.remove(track)
                else:
                    if len(self.selected_tracks) < 2:  # Max 2 tracks for mixing
                        self.selected_tracks.append(track)
        
        # Mixing controls (right side)
        else:
            button_y = 50
            
            # Create Mix button
            if (220 <= rel_x <= 380 and button_y <= rel_y <= button_y + self.button_height):
                self._create_mix()
            
            # Preview Mix button
            button_y += self.button_height + self.button_margin
            if (220 <= rel_x <= 380 and button_y <= rel_y <= button_y + self.button_height):
                self.show_mix_preview = not self.show_mix_preview
            
            # Sell Mix button
            button_y += self.button_height + self.button_margin
            if (220 <= rel_x <= 380 and button_y <= rel_y <= button_y + self.button_height):
                self.show_sales_panel = not self.show_sales_panel
            
            # Clear Selection button
            button_y += self.button_height + self.button_margin
            if (220 <= rel_x <= 380 and button_y <= rel_y <= button_y + self.button_height):
                self.selected_tracks.clear()
                self.current_mix = None
    
    def _handle_key_press(self, ev, inventory: List[AlienTrack]):
        """Handle keyboard input"""
        if ev.key == pygame.K_m:
            # M key to create mix
            self._create_mix()
        elif ev.key == pygame.K_p:
            # P key to preview mix
            self.show_mix_preview = not self.show_mix_preview
        elif ev.key == pygame.K_s:
            # S key to show sales panel
            self.show_sales_panel = not self.show_sales_panel
        elif ev.key == pygame.K_c:
            # C key to clear selection
            self.selected_tracks.clear()
            self.current_mix = None
    
    def _create_mix(self):
        """Create a mix from selected tracks"""
        if len(self.selected_tracks) != 2:
            return
        
        track1, track2 = self.selected_tracks
        
        # Check if tracks can be combined
        if not track1.can_combine_with(track2):
            return
        
        # Create the mix
        self.current_mix = track1.create_mix_with(track2)
        
        # Get daughter's rating
        self.game_state.daughter_rating = self.game_state.get_daughter_rating(
            self.current_mix.descriptors
        )
    
    def render(self, screen, inventory: List[AlienTrack]):
        """Render the mixing station"""
        # Main panel background
        panel_surface = pygame.Surface((self.panel_width, self.panel_height))
        panel_surface.fill((25, 30, 50))
        pygame.draw.rect(panel_surface, UI_OUTLINE, 
                        (0, 0, self.panel_width, self.panel_height), 3)
        
        # Title
        title = self.FONT.render("Mixing Station", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(self.panel_width // 2, 20))
        panel_surface.blit(title, title_rect)
        
        # Left side: Track selection
        self._render_track_selection(panel_surface, inventory)
        
        # Right side: Mixing controls
        self._render_mixing_controls(panel_surface)
        
        # Current mix info
        if self.current_mix:
            self._render_mix_info(panel_surface)
        
        # Sales panel overlay
        if self.show_sales_panel:
            self._render_sales_panel(panel_surface)
        
        # Mix preview overlay
        if self.show_mix_preview and self.current_mix:
            self._render_mix_preview(panel_surface)
        
        # Blit panel to screen
        screen.blit(panel_surface, (self.panel_x, self.panel_y))
    
    def _render_track_selection(self, surface, inventory: List[AlienTrack]):
        """Render the track selection area"""
        # Section title
        title = self.SMALL.render("Select Tracks to Mix:", True, TEXT_COLOR)
        surface.blit(title, (10, 50))
        
        # Track list
        y = 80
        for i, track in enumerate(inventory):
            # Track background
            bg_color = HILITE if track in self.selected_tracks else (60, 60, 80)
            pygame.draw.rect(surface, bg_color, (10, y, 180, 35))
            pygame.draw.rect(surface, UI_OUTLINE, (10, y, 180, 35), 1)
            
            # Track name
            name_text = self.SMALL.render(track.name[:20], True, TEXT_COLOR)
            surface.blit(name_text, (15, y + 8))
            
            # Track value
            value_text = self.SMALL.render(f"${track.value}", True, INFO_COLOR)
            surface.blit(value_text, (15, y + 20))
            
            y += 40
        
        # Instructions
        y += 10
        instructions = [
            "Click tracks to select",
            "Max 2 tracks per mix",
            "Surface + Underground = Best!"
        ]
        for instruction in instructions:
            inst_text = self.SMALL.render(instruction, True, INFO_COLOR)
            surface.blit(inst_text, (10, y))
            y += 18
    
    def _render_mixing_controls(self, surface):
        """Render the mixing controls"""
        # Section title
        title = self.SMALL.render("Mixing Controls:", True, TEXT_COLOR)
        surface.blit(title, (220, 50))
        
        y = 80
        
        # Create Mix button
        button_color = HILITE if len(self.selected_tracks) == 2 else (80, 80, 80)
        pygame.draw.rect(surface, button_color, (220, y, 160, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (220, y, 160, self.button_height), 1)
        
        text = self.SMALL.render("Create Mix (M)", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(300, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        # Preview Mix button
        y += self.button_height + self.button_margin
        button_color = HILITE if self.current_mix else (80, 80, 80)
        pygame.draw.rect(surface, button_color, (220, y, 160, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (220, y, 160, self.button_height), 1)
        
        text = self.SMALL.render("Preview Mix (P)", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(300, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        # Sell Mix button
        y += self.button_height + self.button_margin
        button_color = HILITE if self.current_mix else (80, 80, 80)
        pygame.draw.rect(surface, button_color, (220, y, 160, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (220, y, 160, self.button_height), 1)
        
        text = self.SMALL.render("Sell Mix (S)", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(300, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        # Clear Selection button
        y += self.button_height + self.button_margin
        pygame.draw.rect(surface, (120, 60, 60), (220, y, 160, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (220, y, 160, self.button_height), 1)
        
        text = self.SMALL.render("Clear Selection (C)", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(300, y + self.button_height // 2))
        surface.blit(text, text_rect)
    
    def _render_mix_info(self, surface):
        """Render information about the current mix"""
        y = 300
        
        # Mix title
        title = self.SMALL.render("Current Mix:", True, TEXT_COLOR)
        surface.blit(title, (10, y))
        y += 20
        
        mix_name = self.SMALL.render(self.current_mix.name, True, HILITE)
        surface.blit(mix_name, (15, y))
        y += 25
        
        # Descriptors
        desc_title = self.SMALL.render("Descriptors:", True, TEXT_COLOR)
        surface.blit(desc_title, (10, y))
        y += 20
        
        desc_text = self.SMALL.render(self.current_mix.get_descriptor_summary(), True, INFO_COLOR)
        surface.blit(desc_text, (15, y))
        y += 25
        
        # Daughter's rating
        daughter_title = self.SMALL.render("Daughter's Rating:", True, TEXT_COLOR)
        surface.blit(daughter_title, (10, y))
        y += 20
        
        rating_color = (0, 255, 0) if self.game_state.daughter_rating > 0 else (255, 0, 0)
        rating_text = self.SMALL.render(f"{self.game_state.daughter_rating}/10", True, rating_color)
        surface.blit(rating_text, (15, y))
        y += 25
        
        # Value
        value_title = self.SMALL.render("Estimated Value:", True, TEXT_COLOR)
        surface.blit(value_title, (10, y))
        y += 20
        
        value_text = self.SMALL.render(f"${self.current_mix.value}", True, HILITE)
        surface.blit(value_text, (15, y))
    
    def _render_sales_panel(self, surface):
        """Render the sales panel overlay"""
        if not self.current_mix:
            return
        
        # Create overlay
        overlay = pygame.Surface((350, 400))
        overlay.fill((30, 35, 55))
        pygame.draw.rect(overlay, UI_OUTLINE, (0, 0, 350, 400), 2)
        
        y = 10
        title = self.SMALL.render("Sell Your Mix!", True, TEXT_COLOR)
        overlay.blit(title, (10, y))
        y += 25
        
        # Show potential buyers
        subtitle = self.SMALL.render("Potential Buyers:", True, INFO_COLOR)
        overlay.blit(subtitle, (10, y))
        y += 20
        
        for species_name, species_data in ALIEN_SPECIES.items():
            rating = self.current_mix.species_ratings[species_name]
            
            # Species name
            species_color = species_data["color"]
            species_text = self.SMALL.render(species_name, True, species_color)
            overlay.blit(species_text, (10, y))
            y += 20
            
            # Rating and potential price
            if rating >= 6:
                price = self.current_mix.value * (1 + rating * 0.1)
                price_text = self.SMALL.render(f"Rating: {rating}/10 - Price: ${price:.0f}", True, (0, 255, 0))
                overlay.blit(price_text, (20, y))
            else:
                price_text = self.SMALL.render(f"Rating: {rating}/10 - Not interested", True, (128, 128, 128))
                overlay.blit(price_text, (20, y))
            y += 25
        
        # Daughter's opinion
        y += 10
        daughter_title = self.SMALL.render("Daughter's Opinion:", True, TEXT_COLOR)
        overlay.blit(daughter_title, (10, y))
        y += 20
        
        if self.game_state.daughter_rating >= 7:
            opinion = "This is AMAZING! It'll be a hit!"
            color = (0, 255, 0)
            bonus = " +50% to all prices!"
        elif self.game_state.daughter_rating >= 4:
            opinion = "It's... okay, I guess."
            color = (255, 255, 0)
            bonus = ""
        else:
            opinion = "Ugh, this is terrible. No one will buy it."
            color = (255, 0, 0)
            bonus = " -25% to all prices"
        
        opinion_text = self.SMALL.render(opinion, True, color)
        overlay.blit(opinion_text, (20, y))
        y += 20
        
        if bonus:
            bonus_text = self.SMALL.render(bonus, True, color)
            overlay.blit(bonus_text, (20, y))
            y += 20
        
        # Sell button
        y += 20
        pygame.draw.rect(overlay, HILITE, (10, y, 330, 40))
        pygame.draw.rect(overlay, UI_OUTLINE, (10, y, 330, 40), 1)
        
        sell_text = self.FONT.render("SELL MIX!", True, TEXT_COLOR)
        sell_rect = sell_text.get_rect(center=(175, y + 20))
        overlay.blit(sell_text, sell_rect)
        
        # Blit overlay to main panel
        surface.blit(overlay, (25, 50))
    
    def _render_mix_preview(self, surface):
        """Render a preview of the current mix"""
        if not self.current_mix:
            return
        
        # Create overlay
        overlay = pygame.Surface((350, 300))
        overlay.fill((30, 35, 55))
        pygame.draw.rect(overlay, UI_OUTLINE, (0, 0, 350, 300), 2)
        
        y = 10
        title = self.SMALL.render("Mix Preview", True, TEXT_COLOR)
        overlay.blit(title, (10, y))
        y += 25
        
        # Mix details
        details = [
            f"Name: {self.current_mix.name}",
            f"Duration: {self.current_mix.duration:.1f}s",
            f"Quality: {self.current_mix.recording_quality:.1%}",
            f"Value: ${self.current_mix.value}",
            "",
            "Descriptors:",
        ]
        
        for detail in details:
            if detail:
                detail_text = self.SMALL.render(detail, True, INFO_COLOR)
                overlay.blit(detail_text, (10, y))
                y += 20
            else:
                y += 10
        
        # Show descriptors
        for desc in self.current_mix.descriptors[:8]:  # Limit to 8 descriptors
            desc_text = self.SMALL.render(f"• {desc}", True, TEXT_COLOR)
            overlay.blit(desc_text, (20, y))
            y += 18
        
        # Blit overlay to main panel
        surface.blit(overlay, (25, 50))

