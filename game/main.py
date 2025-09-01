import pygame
import sys
import argparse
from .core.config import WIDTH, HEIGHT, fonts, DEBUG_MODE
from .core.debug import set_debug_mode
from .core.screen import ScreenManager
from .audio.mixer import init_mixer
from .world.overworld_screen import OverworldScreen

def run(auto_close_seconds=0):
    # Enable debug mode if configured
    if DEBUG_MODE:
        # Set up logging to a file in the game directory
        import os
        log_file = os.path.join(os.path.dirname(__file__), "debug.log")
        log_file = os.path.abspath(log_file)
        set_debug_mode(True, log_file)
    
    pygame.init()
    init_mixer()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cosmic Underground – Alien DJ Game")
    FONT, SMALL, MONO = fonts()
    clock = pygame.time.Clock()

    mgr = ScreenManager()
    mgr.push(OverworldScreen((FONT, SMALL, MONO)))

    running = True
    start_time = pygame.time.get_ticks()
    
    # Auto-close timer for testing (0 = disabled)
    AUTO_CLOSE_SECONDS = auto_close_seconds
    
    try:
        while running:
            dt = clock.tick(60) / 1000.0
            
            # Check auto-close timer
            if AUTO_CLOSE_SECONDS > 0:
                elapsed_time = (pygame.time.get_ticks() - start_time) / 1000.0
                if elapsed_time >= AUTO_CLOSE_SECONDS:
                    print(f"🎵 Auto-closing game after {AUTO_CLOSE_SECONDS} seconds for testing")
                    running = False
                    break
            
            # Add a safety timeout to prevent infinite loops
            if dt > 1.0:  # If frame takes more than 1 second, something's wrong
                print(f"⚠️ Warning: Frame took {dt:.2f}s, possible infinite loop detected")
                if dt > 10.0:  # If frame takes more than 10 seconds, force exit
                    print("💥 Frame timeout exceeded 10 seconds, forcing exit")
                    running = False
                    break
            
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                else:
                    mgr.handle_event(ev)

            mgr.update(dt)
            mgr.draw(screen)
            pygame.display.flip()
            
            # Show countdown timer in window title
            if AUTO_CLOSE_SECONDS > 0:
                remaining = max(0, AUTO_CLOSE_SECONDS - elapsed_time)
                pygame.display.set_caption(f"Cosmic Underground – Alien DJ Game (Auto-close in {remaining:.0f}s)")
    finally:
        # Ensure log file is closed properly
        from .core.debug import close_log
        close_log()

    pygame.quit()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cosmic Underground - Alien DJ Game")
    parser.add_argument("--auto-close", "-a", type=int, default=0, 
                       help="Auto-close the game after specified seconds (e.g., --auto-close 30)")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Run the game with auto-close if specified
    if args.auto_close > 0:
        print(f"🎵 Auto-close enabled: Game will close after {args.auto_close} seconds")
        run(auto_close_seconds=args.auto_close)
    else:
        print("🎵 Starting game normally (no auto-close)")
        run()
