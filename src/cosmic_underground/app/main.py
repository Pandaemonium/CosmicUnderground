from cosmic_underground.controller import GameController

def main():
    import pygame
    try:
        GameController().run()
    except SystemExit:
        raise
    except Exception as e:
        print("Fatal error:", e)
    finally:
        # ensure pygame cleans up if anything goes wrong
        try:
            pygame.quit()
        except Exception:
            pass

if __name__ == "__main__":
    import sys, traceback, faulthandler
    faulthandler.enable() 
    try:
        GameController().run()
    except Exception as e:
        traceback.print_exc()   # <-- shows which module/line failed
        print("Fatal error:", e)
        sys.exit(1)
