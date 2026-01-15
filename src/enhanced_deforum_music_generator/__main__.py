"""Package entrypoint.

Examples:
  python -m enhanced_deforum_music_generator ui --port 7860
  python -m enhanced_deforum_music_generator generate <audio.wav> --output out/
  python -m enhanced_deforum_music_generator selfcheck
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    cmd = argv[0] if argv else "selfcheck"
    if cmd in {"selfcheck", "check"}:
        from .cli.selfcheck import run
        return int(run())

    if cmd in {"ui", "generate", "analyze"}:
        # Delegate to the full standalone module (keeps behavior consistent with upstream).
        from . import enhanced_deforum_music_generator as mod
        sys.argv = [sys.argv[0]] + argv
        return int(mod.main())

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
