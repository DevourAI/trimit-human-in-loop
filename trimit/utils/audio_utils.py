from trimit.models import SpeechSegment
import subprocess
import numpy as np
import torch


def seconds_to_samples(segment, sample_rate):
    return [int(segment.start * sample_rate), int(segment.end * sample_rate)]


def extract_waveform_segments(waveform, sample_rate, segments):
    extracted_segments = []
    for segment in segments:
        start_sample, end_sample = seconds_to_samples(segment, sample_rate)
        if len(waveform.shape) == 1:
            segment_waveform = waveform[start_sample:end_sample]
        else:
            segment_waveform = waveform[:, start_sample:end_sample]
        extracted_segments.append(segment_waveform)
    return extracted_segments


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


def load_audio_with_segments(
    audio_file_path: str, segments: list[SpeechSegment], sample_rate=None, flatten=True
):
    waveform, sample_rate = load_audio(audio_file_path, sr=sample_rate, flatten=flatten)

    if segments is not None:
        wave_forms = extract_waveform_segments(waveform, sample_rate, segments)
    else:
        wave_forms = [waveform]
    return wave_forms


def load_multiple_audio_files_as_single_waveform(
    audio_file_paths: list[str],
    sample_rate=None,
    flatten=True,
    to_torch=True,
    buffer_s=0,
):
    if sample_rate is None:
        sample_rate = 16000
    segments = []
    waveform_segments = []
    for audio_file_path in audio_file_paths:
        waveform, _ = load_audio(audio_file_path, flatten=flatten, sr=sample_rate)
        if buffer_s > 0:
            buffer_samples = int(buffer_s * sample_rate)
            zeros = np.zeros(buffer_samples, dtype=waveform.dtype)
            if not flatten:
                zeros = zeros[np.newaxis, :]
            waveform = np.hstack([waveform, zeros])
        waveform_segments.append(waveform)
        if flatten:
            end = waveform.shape[0] / sample_rate
        else:
            end = waveform.shape[1] / sample_rate
        print(f"waveform.shape={waveform.shape}, end={end}")
        segments.append(SpeechSegment(start=0, end=end))
    if len(segments) == 0:
        return None, [], sample_rate
    waveform = np.hstack(waveform_segments)
    if to_torch:
        waveform = torch.from_numpy(waveform)
    return waveform, segments, sample_rate
