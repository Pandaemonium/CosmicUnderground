# Placeholder audio engine façade (we’ll fill this in Phase 5)
import pygame
import numpy as np
import soundfile as sf
import pygame.sndarray as sndarray
from cosmic_underground.core import config as C

class Engine:
    def __init__(self):
        # ensure mixer
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=C.ENGINE_SR, size=-16, channels=16, buffer=2048)
        self.base_channel = 8  # leave lower channels for game/broadcast
        self.track_chans: list[pygame.mixer.Channel] = []
        # clip_id -> { 'chan': int, 'guard': object | None }
        self._voices: dict[int, dict] = {}
        # no loop state; mixer uses absolute timeline only

    def note_seek(self):
        """Called when the user manually seeks the playhead.
        Allows clips to retrigger from the new position by clearing the per-run
        'already played' registry and resetting normalized wrap tracking.
        """
        self._played_in_run.clear()
        self._last_pos_n = None

    def _ensure_channels(self, n_tracks: int):
        need = self.base_channel + max(0, int(n_tracks))
        if pygame.mixer.get_num_channels() < need:
            pygame.mixer.set_num_channels(need)
        self.track_chans = [pygame.mixer.Channel(self.base_channel + i) for i in range(n_tracks)]

    def _stop_all(self):
        for ch in self.track_chans:
            try:
                ch.stop()
            except Exception:
                pass
        self._voices.clear()

    # Public API for Mixer/Controller to immediately stop all DAW audio
    def stop_all(self):
        self._stop_all()

    def update(self, project, transport):
        # init channels per number of tracks
        self._ensure_channels(len(project.tracks))
        if not transport.playing:
            # purge finished voices, keep any still-playing intact
            for cid, entry in list(self._voices.items()):
                ch_id = int(entry.get('chan', -1))
                try:
                    if ch_id < 0 or not pygame.mixer.Channel(ch_id).get_busy():
                        self._voices.pop(cid, None)
                except Exception:
                    self._voices.pop(cid, None)
            return

        # time conversions
        sec_per_bar = transport.seconds_per_bar()
        # absolute timeline (no looping)
        pos_abs = float(transport.pos_bars)

        # Build quick index of clips by id and track for stop pass
        clip_index: dict[int, tuple] = {}
        for ti, t in enumerate(project.tracks):
            for c in t.clips:
                clip_index[getattr(c, 'clip_id', -1)] = (c, ti)

        # PASS 1: remove finished voices based on channel state
        for cid, entry in list(self._voices.items()):
            ch_id = int(entry.get('chan', -1))
            try:
                if ch_id < 0 or not pygame.mixer.Channel(ch_id).get_busy():
                    self._voices.pop(cid, None)
            except Exception:
                self._voices.pop(cid, None)

        # Track occupancy (ensure at most one clip per track at a time)
        track_busy = {ti: False for ti in range(len(project.tracks))}
        for cid, entry in self._voices.items():
            # mark existing active tracks as busy
            chan = int(entry.get('chan', -1))
            ti = chan - self.base_channel
            if 0 <= ti < len(project.tracks):
                track_busy[ti] = True

        # PASS 2: start voices that are now inside
        for ti, t in enumerate(project.tracks):
            if ti >= len(self.track_chans):
                break
            if track_busy.get(ti):
                continue  # already has an active clip
            ch = self.track_chans[ti]
            for c in t.clips:
                cid = getattr(c, 'clip_id', -1)
                if cid in self._voices:
                    continue
                inside, offset_sec, remain_sec = self._inside_and_offset_abs(c, pos_abs, project.timesig[0], sec_per_bar)
                if not inside or remain_sec <= 1e-6:
                    continue
                snd, guard = self._make_offset_sound(c.source_path, max(0.0, offset_sec), max_dur_sec=remain_sec)
                if snd is None:
                    continue
                ch.play(snd, loops=0)
                self._voices[cid] = {
                    'chan': (self.base_channel + ti),
                    'guard': guard,
                }
                track_busy[ti] = True
                break  # one clip per track at a time

    def _inside_and_offset_abs(self, c, pos_bars: float, timesig_n: int, sec_per_bar: float) -> tuple[bool, float, float]:
        """Absolute timeline: inside if start <= pos < start+len.
        Returns (inside, offset_sec, remain_sec).
        """
        beats_per_bar = max(1, int(timesig_n))
        clip_len_bars = float(getattr(c, 'length_beats', 0.0)) / float(beats_per_bar)
        start = float(getattr(c, 'start_bar', 0.0))
        p = float(pos_bars)
        end = start + clip_len_bars
        inside = (p >= start) and (p < end)
        if not inside:
            return False, 0.0, 0.0
        off_bars = max(0.0, p - start)
        rem_bars = max(0.0, end - p)
        return True, off_bars * sec_per_bar, rem_bars * sec_per_bar

    def _make_offset_sound(self, path: str, offset_sec: float, max_dur_sec: float):
        """Create a pygame Sound that starts at offset_sec and lasts up to max_dur_sec.
        Returns (Sound, guard_array) or (None, None) on failure.
        """
        try:
            with sf.SoundFile(path, mode='r') as f:
                sr = int(f.samplerate)
                chs = int(f.channels)
                # compute start frame
                start_frame = int(max(0, offset_sec) * sr)
                if start_frame >= len(f):
                    return None, None
                f.seek(start_frame)
                frames_to_read = int(max(1, round(max_dur_sec * sr)))
                data = f.read(frames=frames_to_read, dtype='float32', always_2d=True)
                if data.size == 0:
                    return None, None
                # resample if needed to engine sample rate
                target_sr = int(C.ENGINE_SR)
                y = data
                if sr != target_sr:
                    dur = y.shape[0] / float(sr)
                    tgt = max(1, int(round(dur * target_sr)))
                    t_in = np.linspace(0.0, dur, y.shape[0], endpoint=False, dtype=np.float64)
                    t_out= np.linspace(0.0, dur, tgt,       endpoint=False, dtype=np.float64)
                    out = np.empty((tgt, y.shape[1]), dtype=np.float32)
                    for c in range(y.shape[1]):
                        out[:, c] = np.interp(t_out, t_in, y[:, c]).astype(np.float32)
                    y = out
                # convert to int16 expected by mixer
                y = np.clip(y, -1.0, 1.0)
                y = np.ascontiguousarray((y * 32767.0).astype(np.int16))
                # If mixer is stereo and source is mono, expand channels
                if y.shape[1] == 1:
                    y = np.repeat(y, 2, axis=1)
                snd = sndarray.make_sound(y)
                return snd, y
        except Exception:
            return None, None
