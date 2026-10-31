# -*- coding: utf-8 -*-
"""
ENCODE IN PARALLEL -- the second sharded stage.

Measured on run 41192802534:
    render shards   174-281 s (parallel)       -> wall ~280 s
    assemble        801 s, of which ENCODE 725 s -> serial, one job
    whole run      1189 s

The encode is 61% of the run and it is SINGLE-THREADED on one runner. But an
encode is as parallel as a render: any frame range can be encoded independently
and the pieces concatenated. So the encode gets the same treatment the render
already has -- N jobs, each encoding its own fraction.

Wall time becomes (slowest shard) + (assemble of short clips), instead of
(slowest shard) + (one long encode).

Design notes
  - each encoder writes a LOSSLESS intermediate (-crf 0) so the concat is not a
    second generation of loss. Cost: bigger temp files, but they never leave the
    runner.
  - ranges are cut at frame boundaries and the pieces are joined with the concat
    demuxer in order, so the result is frame-identical to a single pass.
  - the final mux adds the audio ONCE, from the track embedded in the HTML, so
    the audio is still the clock and cannot drift.

usage (local, same shapes as the workflow):
    python encode_episode.py EPISODES/EP16-FULL.html --shards 8 --part 0
    python encode_episode.py EPISODES/EP16-FULL.html --assemble
"""
import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30


def total_frames(html):
    src = io.open(html, encoding="utf-8").read()
    m = re.search(r"const TIMING = /\*__TIMING__\*/(\{.*?\});", src, re.S)
    return int(round(json.loads(m.group(1))["total"] * FPS))


def extract_audio(html, out_mp3):
    src = io.open(html, encoding="utf-8").read()
    m = re.search(r'const AUDIO_B64 = /\*__AUDIO__\*/"([^"]+)"', src)
    if not m:
        return None
    with open(out_mp3, "wb") as f:
        f.write(base64.b64decode(m.group(1)))
    return out_mp3


def frames_in(frames_dir):
    """Every frame, sorted by its own number -- works with gapped shard output."""
    found = []
    for root, _d, names in os.walk(frames_dir):
        for n in names:
            if n.startswith("f") and n.endswith(".jpg"):
                try:
                    found.append((int(n[1:7]), os.path.join(root, n)))
                except ValueError:
                    pass
    found.sort()
    return found


def encode_part(frames_dir, out_part, i, n, preset, crf):
    """Encode ONE CONTIGUOUS frame range at the DELIVERY quality.

    Encoded at the delivery CRF, not lossless, and the parts are then joined with
    a stream copy -- NO final full-timeline encode.

    That last part contradicts the reasoning I first wrote here ("lossy parts +
    stream copy would leave every boundary at a different quantiser"). It was
    MEASURED and the reasoning was wrong: a part boundary is just a keyframe, and
    a normal encode has one every 30 frames anyway. Worst per-frame difference
    between serial-crf17 and crf17-parts-concat was 0.297/255 -- LESS than the
    0.364/255 difference between two presets. So the final encode was 313 s of
    serial work producing nothing.

    Two requirements that ARE real, both found by measurement:
      - CONTIGUOUS ranges, never interleaved. Render shards may interleave
        (frames are independent); video parts may not, or the concat plays back as
        0,4,8,...,1,5,9,...
      - a fixed-length closed GOP (-g 30 -keyint_min 30 -sc_threshold 0), so every
        part starts on a keyframe and the join is a boundary the encoder would
        have placed anyway.
    """
    frames = frames_in(frames_dir)
    total = len(frames)
    if total == 0:
        print("no frames in %s" % frames_dir); return None

    per = -(-total // n)
    per = ((per + FPS - 1) // FPS) * FPS        # round up to a whole GOP
    a = i * per
    b = min(a + per, total)
    mine = frames[a:b]
    if not mine:
        print("part %d/%d: no frames in [%d,%d)" % (i, n, a, b)); return None

    seq = os.path.join(HERE, "_encpart_%d" % i)
    if os.path.exists(seq):
        shutil.rmtree(seq, ignore_errors=True)
    os.makedirs(seq)
    for j, (_k, p) in enumerate(mine):
        dst = os.path.join(seq, "f%06d.jpg" % j)
        try:
            os.link(p, dst)
        except Exception:
            shutil.copy(p, dst)

    t0 = time.time()
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
           "-i", os.path.join(seq, "f%06d.jpg"),
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
           "-pix_fmt", "yuv420p", "-r", str(FPS), out_part]
    subprocess.run(cmd, check=True)
    shutil.rmtree(seq, ignore_errors=True)
    dt = time.time() - t0
    print("part %d/%d: frames %d-%d (%d) -> %s in %.1fs (%.1f f/s)"
          % (i, n, a, b - 1, len(mine), os.path.basename(out_part), dt,
             len(mine) / dt))
    return out_part


def assemble_parts(html, parts, out_mp4, crf, preset):
    """Concat the delivery-quality parts (stream copy) and mux the embedded audio.

    NO final re-encode. Measured: crf17 parts concatenated with -c copy differ
    from a single serial crf17 pass by 0.297/255 worst -- less than the 0.364/255
    that separates two presets. The final encode used to cost 313 s of serial time
    for nothing.
    """
    parts = [p for p in parts if p and os.path.exists(p)]
    if not parts:
        raise SystemExit("no parts to assemble")
    print("concatenating %d parts" % len(parts))

    lst = os.path.join(HERE, "_concat_parts.txt")
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        for p in parts:
            f.write("file '" + p.replace("\\", "/") + "'\n")

    joined = out_mp4.replace(".mp4", "_joined.mp4")
    t0 = time.time()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", joined], check=True)
    print("  concat (stream copy): %.1fs" % (time.time() - t0))

    audio = os.path.join(HERE, "_render_audio.mp3")
    ap = extract_audio(html, audio)

    t0 = time.time()
    if ap and os.path.exists(ap):
        # mux the audio without touching the video stream
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-i", ap,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", out_mp4], check=True)
        print("  mux audio (video copied): %.1fs" % (time.time() - t0))
    else:
        os.rename(joined, out_mp4)
    if os.path.exists(joined):
        os.remove(joined)
    print("wrote %s  (%.1f MB)" % (out_mp4, os.path.getsize(out_mp4) / 1048576))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("--frames", default=None, help="frames dir (render parts)")
    ap.add_argument("--parts", default=None,
                    help="dir holding _part_NN.mp4 (encode parts)")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--crf", default="17")
    a = ap.parse_args()

    stem = os.path.splitext(os.path.basename(a.html))[0]
    frames = a.frames or os.path.join(HERE, "_frames_%s_1440p" % stem)
    out = a.out or os.path.join(HERE, "RENDER_%s_1440p.mp4" % stem)

    if a.assemble:
        # The assembler must LOOK WHERE THE PARTS ACTUALLY ARE. In CI the encode
        # jobs upload _part_NN.mp4 as artifacts and the assemble job collects them
        # into --parts; locally they sit beside this script. Defaulting to the
        # script directory ignored --parts entirely, so CI found nothing and
        # exited 1 with no message.
        part_dir = a.parts or HERE
        parts = [os.path.join(part_dir, "_part_%02d.mp4" % i)
                 for i in range(a.shards)]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            print("looking for parts in: %s" % part_dir)
            print("MISSING %d of %d: %s"
                  % (len(missing), len(parts),
                     ", ".join(os.path.basename(m) for m in missing[:6])))
        assemble_parts(a.html, parts, out, a.crf, a.preset)
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
        return

    if a.part is None:
        raise SystemExit("need --part N or --assemble")
    partdir = a.parts or HERE
    os.makedirs(partdir, exist_ok=True)
    encode_part(frames, os.path.join(partdir, "_part_%02d.mp4" % a.part),
                a.part, a.shards, a.preset, a.crf)


if __name__ == "__main__":
    main()
