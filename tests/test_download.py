"""URL id resolution must not hit the network for ordinary YouTube links."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soloclip.download import probe_video_id  # noqa: E402


def test_youtu_be_short_link():
    assert probe_video_id("https://youtu.be/BdHK_r9RXTc") == "BdHK_r9RXTc"


def test_watch_url():
    assert probe_video_id("https://www.youtube.com/watch?v=BdHK_r9RXTc") == "BdHK_r9RXTc"


def test_watch_url_with_leading_params():
    assert probe_video_id("https://www.youtube.com/watch?app=desktop&v=BdHK_r9RXTc") == "BdHK_r9RXTc"


def test_id_starting_with_dash():
    """Ids can start with '-', which has bitten this project before."""
    assert probe_video_id("https://youtu.be/-ABCDEFGHIJ") == "-ABCDEFGHIJ"


def test_query_and_fragment_are_ignored():
    assert probe_video_id("https://youtu.be/BdHK_r9RXTc?si=abc&t=90") == "BdHK_r9RXTc"


def test_shorts_and_embed():
    assert probe_video_id("https://www.youtube.com/shorts/BdHK_r9RXTc") == "BdHK_r9RXTc"
    assert probe_video_id("https://www.youtube.com/embed/BdHK_r9RXTc") == "BdHK_r9RXTc"


def test_longer_token_is_not_mistaken_for_an_id():
    """A 12+ char token must fall through to the real resolver, not be truncated."""
    import soloclip.download as dl
    assert dl._YT_ID.search("https://youtu.be/BdHK_r9RXTcXX") is None
