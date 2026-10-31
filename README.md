# byte-forensics-render

Parallel 1440p render for the BYTE FORENSICS episodes.

Each `EPISODES/*.html` is fully self-contained (fonts and narration audio are
embedded, zero external requests). The animation is a paused timeline driven by
`renderAt(t)`, and every animated layer is a pure function of `t` -- so frame
1234 is identical no matter which runner produces it. That makes the render
embarrassingly parallel.

## Run it

Actions -> **render-1440p** -> Run workflow

| input | value |
|---|---|
| episode | `EPISODES/EP16-FULL.html` |
| preset | `1440p` |
| shards | `8` |

Eight shards render disjoint frame ranges concurrently; a final job merges them,
encodes with ffmpeg, muxes the **embedded** audio track (the audio is the clock,
so it must be the same track the animation was built against), and uploads
`RENDER_1440p.mp4` as an artifact.

## Locally

```bash
python render_episode.py EPISODES/EP16-FULL.html --preset 1440p --shard 0/8
python render_episode.py EPISODES/EP16-FULL.html --preset 1440p --assemble
```

`--shard i/N` renders every Nth frame. Run N of them concurrently on N machines
(or just run one of them if you only need a preview range).

## Notes

- Frame capture is JPEG q92, encoded to H.264 CRF 17 -- visually lossless for
  line art on a dark field.
- Output is 2560x1440 at 30fps.
- Nothing from the UI is in the picture: no timecode, no progress bar, no
  controls, no watermark. Only scene content and the captions.
