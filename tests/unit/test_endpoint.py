from __future__ import annotations

import pytest

from main.audio.endpoint import (
    FRAME_MS,
    EndpointConfig,
    Endpointer,
    EndReason,
)

CONFIG = EndpointConfig(
    start_frames=2,
    silence_frames=3,
    no_speech_frames=5,
    max_frames=10,
    pre_roll_frames=2,
)


def test_default_silence_is_three_seconds() -> None:
    assert EndpointConfig().silence_frames == 100


def test_default_waits_are_expressed_in_seconds() -> None:
    frame_s = FRAME_MS / 1000
    config = EndpointConfig()

    # Whole frames: the waits are within one frame of the spec durations.
    assert config.silence_frames * frame_s == pytest.approx(3.0, abs=0.05)
    assert config.no_speech_frames * frame_s == pytest.approx(10.0, abs=0.05)
    assert config.max_frames * frame_s == pytest.approx(30.0, abs=0.05)
    assert config.pre_roll_frames * frame_s == pytest.approx(0.3, abs=0.05)


def test_start_keeps_pre_roll_and_stops_on_silence() -> None:
    endpointer = Endpointer(CONFIG)

    assert endpointer.feed(b"a", False) is None
    assert endpointer.feed(b"b", True) is None
    assert endpointer.feed(b"c", True) is None
    assert endpointer.started
    assert endpointer.audio() == b"bc"

    assert endpointer.feed(b"d", False) is None
    assert endpointer.feed(b"e", False) is None
    assert endpointer.feed(b"f", False) is EndReason.SILENCE
    assert endpointer.audio() == b"bcdef"


def test_no_speech_times_out_with_no_audio() -> None:
    endpointer = Endpointer(CONFIG)

    for _ in range(5):
        reason = endpointer.feed(b"x", False)

    assert reason is EndReason.TIMEOUT
    assert endpointer.audio() == b""


def test_max_duration_stops_recording() -> None:
    config = EndpointConfig(
        start_frames=1,
        silence_frames=100,
        no_speech_frames=100,
        max_frames=4,
        pre_roll_frames=1,
    )
    endpointer = Endpointer(config)

    assert endpointer.feed(b"a", True) is None
    assert endpointer.feed(b"b", True) is None
    assert endpointer.feed(b"c", True) is None
    assert endpointer.feed(b"d", True) is EndReason.MAX_DURATION


def test_frames_after_done_are_ignored() -> None:
    endpointer = Endpointer(CONFIG)

    for _ in range(5):
        endpointer.feed(b"x", False)

    assert endpointer.feed(b"y", True) is None
    assert endpointer.audio() == b""
