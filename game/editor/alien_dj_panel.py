import pygame
from typing import List, Tuple
from ..core.config import (
    WIDTH, HEIGHT, ALIEN_SPECIES, EXEC_PREFERENCES,
    TEXT_COLOR, INFO_COLOR, UI_OUTLINE, HILITE, ERR
)
from ..core.game_state import GameState
from ..audio.alien_track import AlienTrack

class AlienDJPanel:
    def __init__(self, fonts, game_state: GameState):
        self.FONT, self.SMALL, self.MONO = fonts
        self.game_state = game_state
        self.selected_track: AlienTrack = None
        self.show_species_info = False
        self.show_exec_info = False
        self.show_mixing_tips = False
        
        # Panel dimensions
        self.panel_width = 300
        self.panel_height = 400
        self.panel_x = WIDTH - self.panel_width - 20
        self.panel_y = 20
        
        # Button dimensions
        self.button_height = 30
        self.button_margin = 5
        
    def handle_event(self, ev, inventory: List[AlienTrack]):
        """Handle mouse and keyboard events"""
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:  # Left click
                self._handle_mouse_click(ev.pos, inventory)
        elif ev.type == pygame.KEYDOWN:
            self._handle_key_press(ev, inventory)
    
    def _handle_mouse_click(self, pos: Tuple[int, int], inventory: List[AlienTrack]):
        """Handle mouse clicks on the panel"""
        x, y = pos
        
        # Check if click is within panel bounds
        if not (self.panel_x <= x <= self.panel_x + self.panel_width and
                self.panel_y <= y <= self.panel_y + self.panel_height):
            return
        
        # Calculate relative position within panel
        rel_x = x - self.panel_x
        rel_y = y - self.panel_y
        
        # Check button clicks
        button_y = 80  # Start after title and attention bar
        
        # Bootleg button
        if (10 <= rel_x <= 140 and button_y <= rel_y <= button_y + self.button_height):
            if self.game_state.is_recording:
                self.game_state.stop_recording()
            else:
                self.game_state.start_recording()
        
        # Sneak button
        button_y += self.button_height + self.button_margin
        if (10 <= rel_x <= 140 and button_y <= rel_y <= button_y + self.button_height):
            self.game_state.toggle_sneaking()
        
        # Save to lair button
        button_y += self.button_height + self.button_margin
        if (10 <= rel_x <= 140 and button_y <= rel_y <= button_y + self.button_height):
            self.game_state.save_to_lair()
        
        # Toggle info panels
        button_y += self.button_height + self.button_margin
        if (10 <= rel_x <= 140 and button_y <= rel_y <= button_y + self.button_height):
            self.show_species_info = not self.show_species_info
        
        button_y += self.button_height + self.button_margin
        if (10 <= rel_x <= 140 and button_y <= rel_y <= button_y + self.button_height):
            self.show_exec_info = not self.show_exec_info
        
        button_y += self.button_height + self.button_margin
        if (10 <= rel_x <= 140 and button_y <= rel_y <= button_y + self.button_height):
            self.show_mixing_tips = not self.show_mixing_tips
    
    def _handle_key_press(self, ev, inventory: List[AlienTrack]):
        """Handle keyboard input"""
        if ev.key == pygame.K_r:
            # R key to start/stop recording
            if self.game_state.is_recording:
                self.game_state.stop_recording()
            else:
                self.game_state.start_recording()
        elif ev.key == pygame.K_s:
            # S key to toggle sneaking
            self.game_state.toggle_sneaking()
        elif ev.key == pygame.K_l:
            # L key to save to lair
            self.game_state.save_to_lair()
    
    def render(self, screen, inventory: List[AlienTrack]):
        """Render the alien DJ panel"""
        # Main panel background
        panel_surface = pygame.Surface((self.panel_width, self.panel_height))
        panel_surface.fill((20, 25, 45))
        pygame.draw.rect(panel_surface, UI_OUTLINE, 
                        (0, 0, self.panel_width, self.panel_height), 2)
        
        # Title
        title = self.FONT.render("Alien DJ Control Panel", True, TEXT_COLOR)
        panel_surface.blit(title, (10, 10))
        
        # Attention meter
        self._render_attention_meter(panel_surface)
        
        # Control buttons
        self._render_control_buttons(panel_surface)
        
        # Game info
        self._render_game_info(panel_surface, inventory)
        
        # Info panels
        if self.show_species_info:
            self._render_species_info(panel_surface)
        if self.show_exec_info:
            self._render_exec_info(panel_surface)
        if self.show_mixing_tips:
            self._render_mixing_tips(panel_surface)
        
        # Blit panel to screen
        screen.blit(panel_surface, (self.panel_x, self.panel_y))
    
    def _render_attention_meter(self, surface):
        """Render the attention meter"""
        y = 50
        
        # Label
        label = self.SMALL.render("Attention Level:", True, TEXT_COLOR)
        surface.blit(label, (10, y))
        y += 20
        
        # Progress bar background
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(surface, (60, 60, 60), (10, y, bar_width, bar_height))
        
        # Progress bar fill
        attention_pct = self.game_state.get_attention_percentage() / 100.0
        fill_width = int(bar_width * attention_pct)
        attention_color = self.game_state.get_attention_color()
        pygame.draw.rect(surface, attention_color, (10, y, fill_width, bar_height))
        
        # Progress bar outline
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, bar_width, bar_height), 1)
        
        # Percentage text
        pct_text = self.SMALL.render(f"{attention_pct:.1%}", True, TEXT_COLOR)
        surface.blit(pct_text, (220, y))
        
        # Status text
        y += 25
        if self.game_state.attention >= 80:
            status = "DANGER! Getting caught!"
            color = ERR
        elif self.game_state.attention >= 50:
            status = "Warning: High attention"
            color = (255, 165, 0)  # Orange
        else:
            status = "Safe: Low attention"
            color = (0, 255, 0)  # Green
        
        status_text = self.SMALL.render(status, True, color)
        surface.blit(status_text, (10, y))
    
    def _render_control_buttons(self, surface):
        """Render the control buttons"""
        y = 80
        
        # Bootleg button
        button_color = ERR if self.game_state.is_recording else HILITE
        pygame.draw.rect(surface, button_color, (10, y, 130, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, 130, self.button_height), 1)
        
        button_text = "STOP Recording" if self.game_state.is_recording else "START Recording"
        text = self.SMALL.render(button_text, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(75, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        # Sneak button
        y += self.button_height + self.button_margin
        button_color = HILITE if self.game_state.is_sneaking else (100, 100, 100)
        pygame.draw.rect(surface, button_color, (10, y, 130, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, 130, self.button_height), 1)
        
        button_text = "SNEAKING" if self.game_state.is_sneaking else "SNEAK"
        text = self.SMALL.render(button_text, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(75, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        # Save to lair button
        y += self.button_height + self.button_margin
        pygame.draw.rect(surface, HILITE, (10, y, 130, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, 130, self.button_height), 1)
        
        text = self.SMALL.render("Save to Lair", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(75, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        # Info toggle buttons
        y += self.button_height + self.button_margin
        pygame.draw.rect(surface, (80, 80, 120), (10, y, 130, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, 130, self.button_height), 1)
        
        text = self.SMALL.render("Species Info", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(75, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        y += self.button_height + self.button_margin
        pygame.draw.rect(surface, (80, 80, 120), (10, y, 130, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, 130, self.button_height), 1)
        
        text = self.SMALL.render("Exec Preferences", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(75, y + self.button_height // 2))
        surface.blit(text, text_rect)
        
        y += self.button_height + self.button_margin
        pygame.draw.rect(surface, (80, 80, 120), (10, y, 130, self.button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, 130, self.button_height), 1)
        
        text = self.SMALL.render("Mixing Tips", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(75, y + self.button_height // 2))
        surface.blit(text, text_rect)
    
    def _render_game_info(self, surface, inventory: List[AlienTrack]):
        """Render basic game information"""
        y = 280
        
        # Money
        money_text = self.SMALL.render(f"Money: ${self.game_state.money}", True, TEXT_COLOR)
        surface.blit(money_text, (10, y))
        y += 20
        
        # Inventory count
        inventory_text = self.SMALL.render(f"Tracks: {len(inventory)}", True, TEXT_COLOR)
        surface.blit(inventory_text, (10, y))
        y += 20
        
        # Current recording duration
        if self.game_state.is_recording:
            duration_text = self.SMALL.render(
                f"Recording: {self.game_state.current_recording_duration:.1f}s", 
                True, ERR
            )
            surface.blit(duration_text, (10, y))
    
    def _render_species_info(self, surface):
        """Render information about alien species"""
        # Create overlay panel
        overlay = pygame.Surface((280, 300))
        overlay.fill((30, 35, 55))
        pygame.draw.rect(overlay, UI_OUTLINE, (0, 0, 280, 300), 2)
        
        y = 10
        title = self.SMALL.render("Alien Species & Preferences", True, TEXT_COLOR)
        overlay.blit(title, (10, y))
        y += 25
        
        for species_name, species_data in ALIEN_SPECIES.items():
            # Species name
            species_text = self.SMALL.render(species_name, True, species_data["color"])
            overlay.blit(species_text, (10, y))
            y += 20
            
            # Preferences
            prefs_text = self.SMALL.render(
                f"Likes: {', '.join(species_data['preferences'][:3])}", 
                True, INFO_COLOR
            )
            overlay.blit(prefs_text, (20, y))
            y += 20
            
            # Description
            desc_text = self.SMALL.render(species_data["description"], True, INFO_COLOR)
            # Wrap long descriptions
            if desc_text.get_width() > 250:
                desc_text = self.SMALL.render(species_data["description"][:40] + "...", True, INFO_COLOR)
            overlay.blit(desc_text, (20, y))
            y += 25
        
        # Blit overlay to main panel
        surface.blit(overlay, (10, 100))
    
    def _render_exec_info(self, surface):
        """Render executive preferences information"""
        # Create overlay panel
        overlay = pygame.Surface((280, 250))
        overlay.fill((30, 35, 55))
        pygame.draw.rect(overlay, UI_OUTLINE, (0, 0, 280, 250), 2)
        
        y = 10
        title = self.SMALL.render("Executive Preferences", True, TEXT_COLOR)
        overlay.blit(title, (10, y))
        y += 25
        
        subtitle = self.SMALL.render("Satisfy these to get paid!", True, INFO_COLOR)
        overlay.blit(subtitle, (10, y))
        y += 20
        
        for exec_name, preferences in EXEC_PREFERENCES.items():
            # Exec name
            exec_text = self.SMALL.render(exec_name, True, HILITE)
            overlay.blit(exec_text, (10, y))
            y += 20
            
            # Preferences
            prefs_text = self.SMALL.render(
                f"Wants: {', '.join(preferences)}", 
                True, INFO_COLOR
            )
            overlay.blit(prefs_text, (20, y))
            y += 25
        
        # Blit overlay to main panel
        surface.blit(overlay, (10, 100))
    
    def _render_mixing_tips(self, surface):
        """Render mixing tips and advice"""
        # Create overlay panel
        overlay = pygame.Surface((280, 200))
        overlay.fill((30, 35, 55))
        pygame.draw.rect(overlay, UI_OUTLINE, (0, 0, 280, 200), 2)
        
        y = 10
        title = self.SMALL.render("Mixing Tips", True, TEXT_COLOR)
        overlay.blit(title, (10, y))
        y += 25
        
        tips = [
            "• Surface + Underground = Gold!",
            "• Longer recordings = More attention",
            "• Sneak to reduce attention gain",
            "• Save to lair frequently",
            "• Daughter likes unique mixes",
            "• Match species preferences",
            "• Bootlegs are worth more money"
        ]
        
        for tip in tips:
            tip_text = self.SMALL.render(tip, True, INFO_COLOR)
            overlay.blit(tip_text, (10, y))
            y += 20
        
        # Blit overlay to main panel
        surface.blit(overlay, (10, 100))
