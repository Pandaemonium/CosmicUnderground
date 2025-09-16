from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmic_underground.core.world import WorldModel
    from cosmic_underground.audio.service import AudioService
    from cosmic_underground.ui.view import GameView
    from cosmic_underground.mixer.mixer_ui import Mixer
    from cosmic_underground.minigames.dance.engine import DanceMinigame
    from cosmic_underground.controller import GameController


class GameContext:
    """
    A container for shared game objects that various states need to access.
    """
    def __init__(
        self,
        model: WorldModel,
        audio: AudioService,
        view: GameView,
        mixer: Mixer,
        dance_minigame: DanceMinigame,
        controller: "GameController"
    ):
        self.model = model
        self.audio = audio
        self.view = view
        self.mixer = mixer
        self.dance_minigame = dance_minigame
        self.controller = controller
