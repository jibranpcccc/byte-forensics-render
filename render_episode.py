# -*- coding: utf-8 -*-
"""
render_episode.py -- the ONE command that turns an episode HTML into an MP4.

    python render_episode.py EPISODES/EP16-FULL.html --preset 1440p
    python render_episode.py EPISODES/EP16-FULL.html --preset 1440p --shard 0/8
    python render_episode.py EPISODES/EP16-FULL.html --preset 1440p --assemble

Why it looks like this
  The episode is a paused timeline driven by renderAt(t). So a video is simply
  "call renderAt(t) for every frame, screenshot, encode". That is embarrassingly
  parallel: frame N does not depend on frame N-1. The ONLY shared state is the
  HTML file and the audio track.

  Sharding therefore gives near-linear speedup:
      --shard 0/8 ... --shard 7/8   render disjoint frame ranges, concurrently
      --assemble                    concatenate the parts, add the audio, done

  Determinism is already guaranteed by the page (motion is a pure function of t,
  pinned by the gate), so frame 1234 is byte-identical no matter which runner
  produced it.

Audio
  AUDIO_B64 is embedded in the HTML, so the renderer extracts it rather than
  trusting an external file to match. The audio is the clock -- it must be the
  same track the animation was built against.
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "1080p": (1920, 1080, "8M"),
    "1440p": (2560, 1440, "16M"),
    "4k":    (3840, 2160, "32M"),
}
FPS = 30


def extract_audio(html, out_mp3):
    """AUDIO_B64 lives in the HTML so the video and the animation cannot drift."""
    src = io.open(html, encoding="utf-8").read()
    m = re.search(r'const AUDIO_B64 = /\*__AUDIO__\*/"([^"]+)"', src)
    if not m:
        return None
    with open(out_mp3, "wb") as f:
        f.write(base64.b64decode(m.group(1)))
    return out_mp3


def total_seconds(html):
    src = io.open(html, encoding="utf-8").read()
    m = re.search(r"const TIMING = /\*__TIMING__\*/(\{.*?\});\n", src, re.S)
    return json.loads(m.group(1))["total"]


def render(html, width, height, outdir, shard=(0, 1), quality=92, verbose=True):
    from playwright.sync_api import sync_playwright

    os.makedirs(outdir, exist_ok=True)
    total = total_seconds(html)
    n_all = int(round(total * FPS))
    i0, i1 = shard
    mine = [n for n in range(n_all) if n % i1 == i0]
    if verbose:
        print("shard %d/%d: %d of %d frames (%.1fs of video)"
              % (i0, i1, len(mine), n_all, total / i1))

    url = "file:///" + os.path.abspath(html).replace("\\", "/")
    t0 = time.time()
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--force-device-scale-factor=1",
                                     "--hide-scrollbars",
                                     "--disable-lcd-text"])
        pg = br.new_page(viewport={"width": width, "height": height},
                         device_scale_factor=1)
        pg.goto(url, wait_until="load")
        pg.wait_for_function("() => !!window.F && window.F.scenes.length > 0",
                             timeout=120000)
        pg.evaluate("() => { const b=document.getElementById('boot');"
                    "if (b) b.classList.add('hide');"
                    "document.body.classList.add('render'); }")
        pg.wait_for_timeout(400)
        for k, n in enumerate(mine):
            t = n / FPS
            pg.evaluate("(t) => window.renderAt(t)", t)
            # NOTE: Playwright's screenshot(path=...) does NOT expand a printf
            # pattern -- passing "f%06d.jpg" writes ONE file, overwritten every
            # frame. The name must be formatted here.
            pg.screenshot(path=os.path.join(outdir, "f%06d.jpg" % n),
                          type="jpeg", quality=quality)
            if verbose and (k % 120 == 0):
                el = time.time() - t0
                done = k + 1
                rate = done / el if el else 0
                left = (len(mine) - done) / rate / 60 if rate else 0
                print("   %d/%d  %.1f f/s  ~%.1f min left"
                      % (done, len(mine), rate, left), flush=True)
        br.close()
    dt = time.time() - t0
    if verbose:
        print("shard %d/%d done: %d frames in %.1f min (%.2f f/s)"
              % (i0, i1, len(mine), dt / 60, len(mine) / dt))
    return outdir


def assemble(html, outdir, out_mp4, width, height, bitrate, audio_only=False):
    """Encode the frames, then mux the embedded audio. Audio is the clock."""
    audio = os.path.join(HERE, "_render_audio.mp3")
    ap = extract_audio(html, audio)
    total = total_seconds(html)

    frames = sorted(f for f in os.listdir(outdir) if f.endswith(".jpg"))
    if not frames:
        raise SystemExit("no frames in %s" % outdir)
    print("assembling %d frames -> %s" % (len(frames), out_mp4))

    silent = out_mp4.replace(".mp4", "_silent.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-framerate", str(FPS),
           "-i", os.path.join(outdir, "f%06d.jpg"),
           "-c:v", "libx264", "-preset", "medium", "-crf", "17",
           "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-b:v", bitrate, silent]
    subprocess.run(cmd, check=True)

    if ap and os.path.exists(ap):
        cmd2 = ["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", ap,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", out_mp4]
        subprocess.run(cmd2, check=True)
        os.remove(silent)
        print("muxed audio from the embedded track")
    else:
        os.rename(silent, out_mp4)
        print("WARNING: no embedded audio found; wrote a silent video")
    print("wrote %s  (%.1f MB)" % (out_mp4, os.path.getsize(out_mp4) / 1048576))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("--preset", default="1440p", choices=list(PRESETS))
    ap.add_argument("--shard", default="0/1", help="i/N -- render every Nth frame")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    w, h, br = PRESETS[a.preset]
    stem = os.path.splitext(os.path.basename(a.html))[0]
    i, n = (int(x) for x in a.shard.split("/"))

    frames_dir = a.outdir or os.path.join(HERE, "_frames_%s_%s" % (stem, a.preset))
    out_mp4 = a.out or os.path.join(HERE, "RENDER_%s_%s.mp4" % (stem, a.preset))

    if a.assemble:
        assemble(a.html, frames_dir, out_mp4, w, h, br)
    else:
        render(a.html, w, h, frames_dir, shard=(i, n), quality=a.quality)


if __name__ == "__main__":
    main()
