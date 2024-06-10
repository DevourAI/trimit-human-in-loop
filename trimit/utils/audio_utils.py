import subprocess
import numpy as np


def load_audio(file: str, sr: int | None = None, flatten: bool = True):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype, and sample_rate
    """
    from whisperx.audio import SAMPLE_RATE

    if sr is None:
        sr = SAMPLE_RATE
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    buffer = np.frombuffer(out, np.int16)
    if not flatten:
        buffer = buffer[np.newaxis, :]
    return buffer.astype(np.float32) / 32768.0, sr
