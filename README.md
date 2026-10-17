# Real-Time Video Agents

A video agent that makes first-person footage from Meta Ray-Ban glasses
searchable and answerable. One walk-through of a room goes in; out comes a
timeline of short windows, each described by NVIDIA Cosmos Reason for a robot
(objects, spatial relations, people, actions, camera motion), tagged by YOLO11
tracking, embedded for semantic search, and queryable through an agent that
re-watches the relevant windows to answer free-form questions.

Built for the Real-Time Video Agents Hack, London (VAST Builders Challenge).

## What it does

1. **Ingest** - frames of a glasses scan are pulled from the object store and
   cut into 4 s windows (an mp4 clip and a thumbnail per window).
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
5. **Ask** - `/api/scan/{scan}/ask` retrieves the top windows for the question
   and asks Cosmos Reason about their frames, with the window descriptions as
   context, so the answer cites times in seconds.
6. **Page** - a dark, mobile-friendly page with the timeline, search box and
   ask box.

## Architecture

```
glasses app -> object store (R2 today; VAST AI OS in production)
                    |  list_frames / fetch_frame  (the only storage-facing code)
                    v
   ingest (CPU)  -> windows: ffmpeg clips + thumbs
                 -> Reasoner (H100): Cosmos Reason via vLLM + YOLO11 tracking
                 -> MiniLM embeddings -> index.json on a Modal Volume
                    |
   web (CPU, FastAPI) -> timeline / search / ask / media
                                 ask -> Reasoner.ask (H100, on demand)
```

Everything runs on [Modal](https://modal.com): GPU containers scale from zero,
have timeouts and no warm minimum, and the weights are cached on a Volume.

**Where VAST goes.** `list_frames` and `fetch_frame` in `app.py` are the only
functions that touch storage. Today they call a small ingest API in front of
Cloudflare R2 (we had no VAST credentials during the hack). A VAST AI OS
deployment replaces those two functions with S3-compatible calls (boto3 against
the VAST endpoint), and could push new scans into `ingest` from a VAST event
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
modal secret create hf-token    HF_TOKEN=<hugging face read token>
modal secret create replica-live INGEST_TOKEN=<bearer token of your ingest API>
modal secret create rtva-config INGEST_URL=<https://your-ingest-api> \
    DEMO_SCAN=<scan id shown as "demo"> ALLOWED_SCANS=<comma list of scan ids allowed to be indexed>
```

The ingest API is expected to answer `GET /scans/{id}` with
`{"keys": [...], "cursor": ...}` and `GET /scans/{id}/file?key=frames/<name>.jpg`
with the frame bytes; frame names are device time in microseconds.

```
modal deploy app.py                      # web page + endpoints
modal run app.py                         # index the demo scan (about 5 min, one H100)
modal run app.py --scan <id> --force     # index another allowed scan
```

`modal deploy` prints the public URL. Scans are shown by alias (`demo`,
`scan-1`, ...); raw ids never reach the page. `POST /api/ingest/{alias}` needs
`Authorization: Bearer <INGEST_TOKEN>`; everything else is read-only.

## Endpoints

- `GET /` page
- `GET /api/scans` indexed scan aliases
- `GET /api/scan/{alias}` timeline with descriptions and YOLO tags
- `GET /api/scan/{alias}/search?q=` top windows for a query
- `POST /api/scan/{alias}/ask` `{"q": "..."}` agent answer plus the windows it looked at
- `GET /media/{alias}/clips/000.mp4`, `/media/{alias}/thumbs/000.jpg`
