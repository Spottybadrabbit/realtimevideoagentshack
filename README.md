# Real-Time Video Agents

A video agent that makes first-person footage from Meta Ray-Ban glasses
searchable and answerable. One walk-through of a room goes in; out comes a
timeline of short windows, each described by NVIDIA Cosmos Reason for a robot
(objects, spatial relations, people, actions, camera motion), tagged by YOLO11
tracking, embedded for semantic search, and queryable through an agent that
re-watches the relevant windows to answer free-form questions. For scans that
were also reconstructed in 3D, the page embeds the Rerun viewer of the scene
and every result says which frame and camera pose it was filmed at.

Built for the Real-Time Video Agents Hack, London (VAST Builders Challenge).

## What it does

1. **Ingest** - frames of a scan are pulled from the source and cut into 4 s
   windows (an mp4 clip and a thumbnail per window).
2. **Reason** - every window is sent to Cosmos Reason (vLLM on an H100, 4 fps
   as the model was trained) with a prompt that asks for a robot-oriented
   description: what objects are where, what people are doing, notable events,
   how the camera moves.
3. **Detect** - YOLO11 with ByteTrack runs over all frames; each window gets
   object classes with the number of distinct tracks seen.
4. **Search** - window descriptions are embedded with MiniLM; a query such as
   "where is the laptop" or "when did someone open the door" returns the best
   windows with timestamps, thumbnail and clip. YOLO tags give a small boost
   when the query names a detected class.
5. **Ask** - `/api/scan/{alias}/ask` retrieves the top windows for the question
   and asks Cosmos Reason about their frames, with the window descriptions as
   context, so the answer cites times in seconds.
6. **Video in Rerun** - for reconstruction jobs the page embeds the Rerun web
   viewer with the job's `scene.rrd`, and each window carries the frame index
   and camera position from `camera_path.json`.
7. **Page** - a dark, mobile-friendly page with the timeline, search box, ask
   box and the Rerun viewer.

## Sources

| Source | Where the frames come from |
|--------|----------------------------|
| `scan:<id>` | an object store behind a small ingest API (Cloudflare R2 today, VAST AI OS in production). Frames are JPEGs named by device time in microseconds. |
| `job:<id>` | a Replica reconstruction job on the Modal Volume `da3-jobs` (mounted read-only): `frames/*.jpg` at `manifest.json` fps, `camera_path.json` poses, `scene.rrd` for Rerun. Splat-only jobs that kept no source frames are windowed from their rendered fly-through video. |

## Architecture

```
glasses app -> object store (R2 today; VAST AI OS in production)   Replica reconstruction jobs
                    |  list_frames / fetch_frame                          |  pull_job (Volume)
                    v                                                     v
   ingest (CPU)  -> frames named by time -> windows: ffmpeg clips + thumbs
                 -> Reasoner (H100): Cosmos Reason via vLLM + YOLO11 tracking
                 -> MiniLM embeddings (+ camera pose per window) -> index.json on a Modal Volume
                    |
   web (CPU, FastAPI) -> timeline / search / ask / media / Rerun viewer URL
                                 ask -> Reasoner.ask (H100, on demand)
```

Everything runs on [Modal](https://modal.com): GPU containers scale from zero,
have timeouts, no warm minimum and at most one Reasoner container; the weights
are cached on a Volume.

**Where VAST goes.** `list_frames` and `fetch_frame` in `app.py` are the only
functions that touch the object store. Today they call a small ingest API in
front of Cloudflare R2 (we had no VAST credentials during the hack). A VAST AI
OS deployment replaces those two functions with S3-compatible calls (boto3
against the VAST endpoint), and could trigger `ingest` from a VAST event
notification so indexing happens as the footage lands.

## Models

| Role | Model |
|------|-------|
| Video reasoning | `nvidia/Cosmos-Reason1-7B` (vLLM, bf16). `nvidia/Cosmos-Reason2-8B` is a one-line swap once its licence is accepted on Hugging Face. |
| Detection and tracking | `yolo11m.pt` (Ultralytics YOLO11, ByteTrack) |
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

## Deploy

Requires a Modal account and the `modal` CLI.

Secrets (create them in Modal; placeholders below):

```
modal secret create hf-token     HF_TOKEN=<hugging face read token>
modal secret create replica-live INGEST_TOKEN=<bearer token of your ingest API>
modal secret create rtva-config \
    INGEST_URL=<https://your-ingest-api> \
    SCENE_API_URL=<https://your-scene-api>  # serves /jobs/<job>/files/scene.rrd for Rerun; optional
    DEMO_SCAN=<alias shown first> \
    ALLOWED_SCANS=<alias>=job:<job id>,<alias>=scan:<scan id>,...
```

`ALLOWED_SCANS` is the allowlist of what may be indexed, and the public page
only ever sees the aliases. The ingest API is expected to answer
`GET /scans/{id}` with `{"keys": [...], "cursor": ...}` and
`GET /scans/{id}/file?key=frames/<name>.jpg` with the frame bytes.

```
modal deploy app.py                       # web page + endpoints; prints the public URL
modal run app.py                          # index DEMO_SCAN (about 5 min, one H100)
modal run app.py --scan <alias> --force   # index or re-index another allowed scan
```

## Endpoints

- `GET /` page
- `GET /api/scans` indexed scan aliases
- `GET /api/scan/{alias}` timeline with descriptions, YOLO tags, poses and (jobs) `rerun_url`
- `GET /api/scan/{alias}/search?q=` top windows for a query
- `POST /api/scan/{alias}/ask` `{"q": "..."}` agent answer plus the windows it looked at
- `GET /media/{alias}/clips/000.mp4`, `/media/{alias}/thumbs/000.jpg`
- `POST /api/ingest/{alias}` needs `Authorization: Bearer <INGEST_TOKEN>`

## Public demo and limits

The two demo scans are Replica reconstruction jobs `23e1b0194edf` (`lounge`,
a 5 fps walk-through) and `0f2a320f22bf` (`lounge-splat`, windowed from the
Gaussian-splat fly-through). Their job ids are intentionally public: the Rerun
viewer loads each job's `scene.rrd` straight from the scene API.

`/ask` runs the VLM on an H100, so it is bounded: questions are capped at 300
characters, each client IP gets 10 questions per 10 minutes (60 in total,
keyed on the edge-appended client address), and the Reasoner
runs one container with one request at a time. Model output is stripped of
HTML on the server and escaped again on the page.
