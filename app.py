"""realtime-video-agents: make first-person video searchable and answerable.

Pipeline (one `modal run app.py --scan <alias>`):
  frames from a source (object-store scan or reconstruction job)
  -> 4 s windows (mp4 + thumb)
  -> Cosmos Reason (vLLM, H100) describes each window for a robot
  -> YOLO11 tracking tags each window with object classes and counts
  -> MiniLM embeddings of the descriptions -> index.json on a Volume
Then `web` serves a timeline, semantic search, an /ask agent endpoint and,
for reconstruction jobs, the Rerun viewer of the 3D scene.

Sources (configured in the rtva-config secret, see README):
  scan:<id>  frames in an object store behind an ingest API. list_frames /
             fetch_frame are the only functions that talk to it; swap them
             for VAST AI OS (S3-compatible, boto3) and nothing else changes.
             Cloudflare R2 stands in for VAST because we have no VAST creds.
  job:<id>   a Replica reconstruction job on the "da3-jobs" Volume (read-only):
             frames/*.jpg at manifest fps, camera_path.json poses, scene.rrd.
"""

import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

APP = "realtime-video-agents"
# nvidia/Cosmos-Reason2-8B is gated behind the NVIDIA Open Model License. Accept
# it at https://huggingface.co/nvidia/Cosmos-Reason2-8B and change this id;
# Reason2 is Qwen3-VL based and uses the same vLLM path.
VLM_ID = "nvidia/Cosmos-Reason1-7B"
YOLO_WEIGHTS = "/root/.cache/huggingface/yolo11m.pt"
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERUN_VIEWER = "https://app.rerun.io/version/0.37.2/index.html"
WINDOW_S = 4.0  # seconds per window
VLM_FPS = 4  # Cosmos Reason was trained at 4 fps
VIDEO_FPS = 8  # frame rate when a job only has a rendered video, no frame files
DATA = Path("/data")
JOBS = Path("/jobs")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,80}$")
NAME_RE = re.compile(r"^\d{3}\.(mp4|jpg)$")
ASK_LIMIT, ASK_GLOBAL_LIMIT, ASK_PERIOD_S = 10, 60, 600  # per client IP, in total, window

app = modal.App(APP)
data_vol = modal.Volume.from_name("rtva-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("da3-jobs").with_mount_options(read_only=True)
VOLUMES = {DATA.as_posix(): data_vol, "/root/.cache/huggingface": hf_vol}
SECRETS = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("replica-live"),
           modal.Secret.from_name("rtva-config")]
ENV = {"HF_HOME": "/root/.cache/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"}

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("requests", "fastapi[standard]", "sentence-transformers", "numpy")
    .env(ENV)
)
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "vllm==0.21.0", "qwen-vl-utils==0.0.14", "ultralytics==8.4.180", "opencv-python-headless"
    )
    .env(ENV)
)


# ---------------------------------------------------------------- config (from secrets, inside containers)
def sources() -> dict[str, tuple[str, str]]:
    """alias -> (kind, id). ALLOWED_SCANS="lounge=job:abc123,hall=scan:xyz". Only these can be indexed."""
    out = {}
    for item in os.environ.get("ALLOWED_SCANS", "").split(","):
        if "=" not in item:
            continue
        alias, ref = item.strip().split("=", 1)
        kind, _, sid = ref.partition(":")
        if ID_RE.match(alias) and kind in ("scan", "job") and ID_RE.match(sid):
            out[alias] = (kind, sid)
    return out


def resolve(alias: str) -> tuple[str, str]:
    if alias not in sources():
        raise KeyError(alias)
    return sources()[alias]


# ---------------------------------------------------------------- source: object store via ingest API
def _get(path: str, **params):
    import requests

    r = requests.get(
        f"{os.environ['INGEST_URL']}{path}",
        params=params,
        headers={"Authorization": f"Bearer {os.environ['INGEST_TOKEN']}", "User-Agent": f"{APP}/0.1"},
        timeout=60,
    )
    r.raise_for_status()
    return r


def list_frames(scan: str) -> list[str]:
    keys, cursor = [], None
    while True:
        d = _get(f"/scans/{scan}", **({"cursor": cursor} if cursor else {})).json()
        keys += d["keys"]
        cursor = d.get("cursor")
        if not cursor:
            break
    pre = f"scans/{scan}/frames/"
    return sorted(k[len(pre):] for k in keys if k.startswith(pre) and k.endswith(".jpg"))


def fetch_frame(scan: str, name: str, dest: Path):
    if not dest.exists():
        dest.write_bytes(_get(f"/scans/{scan}/file", key=f"frames/{name}").content)


def pull_scan(scan: str, fdir: Path) -> dict:
    """Frames are already named by device time in microseconds."""
    names = list_frames(scan)
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda n: fetch_frame(scan, n, fdir / n), names))
    return {"n_frames": len(names)}


# ---------------------------------------------------------------- source: reconstruction job on a Volume
def pull_job(job: str, fdir: Path) -> dict:
    """Copy a job's frames into fdir named <microseconds>.jpg so windowing is source-agnostic."""
    jdir = JOBS / job
    manifest = json.loads((jdir / "manifest.json").read_text())
    frames = sorted((jdir / "frames").glob("*.jpg"))
    meta = {"fps": manifest.get("fps"), "place": None}
    if (jdir / "place.json").exists():
        meta["place"] = json.loads((jdir / "place.json").read_text()).get("name")
    if frames:
        fps = float(manifest["fps"])
        for i, f in enumerate(frames):
            shutil.copy(f, fdir / f"{int(i / fps * 1e6):016d}.jpg")
    else:  # splat jobs keep only a rendered fly-through video
        video = next(iter(sorted((jdir / "gs" / "gs_video").glob("*.mp4"))), None) or jdir / "source_frames.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps={VIDEO_FPS}",
                        str(fdir / "tmp_%06d.jpg")], check=True)
        for i, f in enumerate(sorted(fdir.glob("tmp_*.jpg"))):
            f.rename(fdir / f"{int(i / VIDEO_FPS * 1e6):016d}.jpg")
        meta["fps"] = VIDEO_FPS
        meta["rendered"] = video.name
    cam = jdir / "camera_path.json"
    if cam.exists() and frames:
        ext = json.loads(cam.read_text())["extrinsics"]  # 4x4 row-major camera poses, one per frame
        meta["poses"] = [[round(m[3], 2), round(m[7], 2), round(m[11], 2)] for m in ext]
    meta["n_frames"] = len(list(fdir.glob("*.jpg")))
    return meta


# ---------------------------------------------------------------- windows
def build_windows(sdir: Path) -> list[dict]:
    """Cut the frame sequence into WINDOW_S windows; write a clip and a thumb per window."""
    fdir = sdir / "frames"
    names = sorted(p.name for p in fdir.glob("*.jpg"))
    t0 = int(names[0][:16])
    ts = [(int(n[:16]) - t0) / 1e6 for n in names]  # frame names are time in microseconds
    bounds, i = [], 0
    while i < len(names):
        j = i
        while j < len(names) and ts[j] < ts[i] + WINDOW_S:
            j += 1
        bounds.append((i, j))
        i = j
    if len(bounds) > 1 and ts[bounds[-1][1] - 1] - ts[bounds[-1][0]] < WINDOW_S / 2:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])  # fold a short tail into the previous window
        bounds.pop()
    (sdir / "clips").mkdir(exist_ok=True)
    (sdir / "thumbs").mkdir(exist_ok=True)
    wins = []
    for k, (i, j) in enumerate(bounds):
        last_dt = ts[j - 1] - ts[j - 2] if j - i > 1 else 1 / 24
        start, end = ts[i], ts[j - 1] + last_dt
        sample, nxt = [], start
        for n, t in zip(names[i:j], ts[i:j]):
            if t >= nxt:
                sample.append(n)
                nxt += 1 / VLM_FPS
        lst = sdir / "clips" / f"{k:03d}.txt"
        lines = [f"file '{fdir / n}'\nduration {ts[a + 1] - ts[a] if a + 1 < j else last_dt:.4f}"
                 for a, n in zip(range(i, j), names[i:j])]
        lst.write_text("\n".join(lines) + f"\nfile '{fdir / names[j - 1]}'\n")
        clip = sdir / "clips" / f"{k:03d}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-vf", "scale=-2:640,format=yuv420p", "-fps_mode", "vfr", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(clip)],
            check=True,
        )
        lst.unlink()
        mid = (i + j) // 2
        shutil.copy(fdir / names[mid], sdir / "thumbs" / f"{k:03d}.jpg")
        wins.append(dict(id=k, start=round(start, 2), end=round(end, 2), frames=[i, j], mid_frame=mid, sample=sample))
    return wins


def embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_ID).encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------- GPU: Cosmos Reason + YOLO
SYSTEM = "You are a helpful assistant."
DESCRIBE = (
    "This {dur:.1f} s clip is first-person video from smart glasses, starting {start:.0f} s into a "
    "walk-through of a space. Describe it for a robot that will later work here. Cover: the objects "
    "visible and their spatial relations (left/right/in front/on/under/next to), any people and what "
    "they are doing, actions and notable events (doors or drawers opening, things picked up or put "
    "down), and how the camera moves. Be concrete and brief: one paragraph, at most 120 words. Do not "
    "speculate about things you cannot see."
)
ASK = (
    "The video is a sequence of first-person clips from smart glasses, in time order, taken at these "
    "moments of a walk-through:\n{ctx}\n\nQuestion: {q}\nAnswer from what you see in the video, name "
    "the time in seconds (e.g. 'at 12 s') when relevant, and be brief."
)


# max_containers=1 and the default of one input at a time bound the GPU cost of the public /ask route.
@app.cls(image=gpu_image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=1800, scaledown_window=240,
         max_containers=1)
class Reasoner:
    @modal.enter()
    def load(self):
        from transformers import AutoProcessor
        from ultralytics import YOLO
        from vllm import LLM

        self.llm = LLM(model=VLM_ID, max_model_len=16384, limit_mm_per_prompt={"video": 1, "image": 0},
                       gpu_memory_utilization=0.85, enforce_eager=True)
        self.processor = AutoProcessor.from_pretrained(VLM_ID)
        self.yolo = YOLO(YOLO_WEIGHTS)
        hf_vol.commit()

    def _generate(self, requests: list[tuple[list[str], str]], max_tokens: int) -> list[str]:
        """requests: (frame file paths, user text) -> answers. Follows the Cosmos-Reason1 vLLM example."""
        from qwen_vl_utils import process_vision_info
        from vllm import SamplingParams

        inputs = []
        for frames, text in requests:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "video", "video": [f"file://{f}" for f in frames], "fps": VLM_FPS},
                    {"type": "text", "text": text},
                ]},
            ]
            prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            _, video_inputs, video_kwargs = process_vision_info(msgs, return_video_kwargs=True)
            inputs.append({"prompt": prompt, "multi_modal_data": {"video": video_inputs},
                           "mm_processor_kwargs": video_kwargs})
        params = SamplingParams(temperature=0.3, top_p=0.95, repetition_penalty=1.05, max_tokens=max_tokens)
        outs = self.llm.generate(inputs, sampling_params=params)
        texts = []
        for o in outs:
            t = o.outputs[0].text
            t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
            m = re.search(r"<answer>(.*?)(</answer>|$)", t, flags=re.S)
            t = m.group(1) if m else t
            texts.append(re.sub(r"<[^>]*>", "", t).strip())  # model output is rendered on a page: no tags
        return texts

    @modal.method()
    def analyse(self, sid: str, wins: list[dict]) -> dict:
        data_vol.reload()
        fdir = DATA / sid / "frames"
        names = sorted(p.name for p in fdir.glob("*.jpg"))

        t = time.time()
        per_frame: dict[str, list] = {}
        for r in self.yolo.track(source=str(fdir), stream=True, persist=True, verbose=False, imgsz=640, conf=0.35):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
            per_frame[Path(r.path).name] = [(r.names[int(c)], i) for c, i in zip(r.boxes.cls.tolist(), ids)]
        for w in wins:
            seen: dict[str, set] = {}
            for n in names[w["frames"][0]:w["frames"][1]]:
                for cls, tid in per_frame.get(n, []):
                    seen.setdefault(cls, set()).add(tid)
            w["tags"] = sorted(([c, len(s)] for c, s in seen.items()), key=lambda x: -x[1])
        yolo_s = time.time() - t

        t = time.time()
        reqs = [([str(fdir / n) for n in w["sample"]], DESCRIBE.format(dur=w["end"] - w["start"], start=w["start"]))
                for w in wins]
        for w, text in zip(wins, self._generate(reqs, max_tokens=320)):
            w["text"] = text
        return {"windows": wins, "yolo_s": round(yolo_s, 1), "vlm_s": round(time.time() - t, 1)}

    @modal.method()
    def ask(self, sid: str, question: str, wins: list[dict]) -> str:
        data_vol.reload()
        fdir = DATA / sid / "frames"
        frames = [str(fdir / n) for w in wins for n in w["sample"][::2]]
        ctx = "\n".join(f"- {w['start']:.0f}-{w['end']:.0f} s: {w['text']}" for w in wins)
        return self._generate([(frames, ASK.format(ctx=ctx, q=question))], max_tokens=400)[0]


# ---------------------------------------------------------------- ingest
@app.function(image=cpu_image, volumes={**VOLUMES, JOBS.as_posix(): jobs_vol}, secrets=SECRETS, timeout=3600)
def ingest(alias: str | None = None, force: bool = False) -> dict:
    alias = alias or os.environ["DEMO_SCAN"]
    kind, sid = resolve(alias)  # KeyError if not in ALLOWED_SCANS
    sdir = DATA / sid
    index = sdir / "index.json"
    if index.exists() and not force:
        return {"alias": alias, "status": "exists"}
    t0 = time.time()
    fdir = sdir / "frames"
    if force:
        shutil.rmtree(sdir, ignore_errors=True)
    fdir.mkdir(parents=True, exist_ok=True)
    meta = pull_scan(sid, fdir) if kind == "scan" else pull_job(sid, fdir)
    if not meta["n_frames"]:
        raise ValueError(f"no frames for {alias}")
    wins = build_windows(sdir)
    data_vol.commit()
    res = Reasoner().analyse.remote(sid, wins)
    wins = res["windows"]
    for w, e in zip(wins, embed([w["text"] for w in wins])):
        w["emb"] = e
        if meta.get("poses"):
            w["pose"] = meta["poses"][w["mid_frame"]]
    doc = {"kind": kind, "model": VLM_ID, "n_frames": meta["n_frames"], "fps": meta.get("fps"),
           "place": meta.get("place"), "rendered": meta.get("rendered"), "duration": wins[-1]["end"],
           "windows": wins, "timing": {"yolo_s": res["yolo_s"], "vlm_s": res["vlm_s"], "total_s": round(time.time() - t0)}}
    index.write_text(json.dumps(doc))
    data_vol.commit()
    return {"alias": alias, "status": "ok", "windows": len(wins), **doc["timing"]}


@app.local_entrypoint()
def main(scan: str = None, force: bool = False):
    print(ingest.remote(scan, force))


# ---------------------------------------------------------------- web
@app.function(image=cpu_image, volumes=VOLUMES, secrets=SECRETS, timeout=600, scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import hmac

    import numpy as np
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel, Field
    from sentence_transformers import SentenceTransformer

    api = FastAPI()
    encoder = SentenceTransformer(EMBED_ID)
    cache: dict = {}
    asks: dict[str, deque] = defaultdict(deque)  # ponytail: in-memory per-container limiter; fine at 1-2 web containers

    def source(alias: str) -> tuple[str, str]:
        try:
            return resolve(alias)
        except KeyError:
            raise HTTPException(404, "unknown scan")

    last_reload = [0.0]

    def fresh():
        """Re-read the Volume so deletions and new ingests show up on every container (at most every 10 s)."""
        if time.time() - last_reload[0] > 10:
            data_vol.reload()
            last_reload[0] = time.time()
            cache.clear()

    def load(alias: str) -> dict:
        kind, sid = source(alias)
        fresh()
        if sid not in cache:
            p = DATA / sid / "index.json"
            if not p.exists():
                raise HTTPException(404, "scan not indexed")
            cache[sid] = json.loads(p.read_text())
        return cache[sid]

    def public(w: dict) -> dict:
        return {k: v for k, v in w.items() if k not in ("emb", "sample", "frames")}

    def public_doc(doc: dict, alias: str, windows: list[dict]) -> dict:
        kind, sid = source(alias)
        out = {k: doc[k] for k in ("kind", "model", "n_frames", "fps", "place", "rendered", "duration", "timing")}
        out.update(scan=alias, windows=windows)
        if kind == "job" and os.environ.get("SCENE_API_URL"):
            # Reconstruction jobs are public by design: the Rerun viewer needs the .rrd URL.
            from urllib.parse import quote

            rrd = f"{os.environ['SCENE_API_URL']}/jobs/{sid}/files/scene.rrd"
            out["rerun_url"] = f"{RERUN_VIEWER}?url={quote(rrd, safe='')}"
        return out

    def search(doc: dict, q: str, k: int = 5) -> list[dict]:
        qv = encoder.encode([q], normalize_embeddings=True)[0]
        words = set(re.findall(r"[a-z]+", q.lower()))
        E = np.array([w["emb"] for w in doc["windows"]])
        sims = E @ qv
        out = []
        for w, s in zip(doc["windows"], sims):
            bonus = 0.15 if any(c.lower() in words or c.lower().rstrip("s") in words for c, _ in w["tags"]) else 0
            out.append((float(s) + bonus, w))
        out.sort(key=lambda x: -x[0])
        return [dict(public(w), score=round(s, 3)) for s, w in out[:k]]

    def rate_limit(request: Request):
        # Rightmost X-Forwarded-For entry is the one Modal's edge appended; leftmost values are client-controlled.
        xff = request.headers.get("x-forwarded-for", "")
        ip = xff.rsplit(",", 1)[-1].strip() or (request.client.host if request.client else "?")
        now = time.time()
        for key in [k for k, q in asks.items() if not q or now - q[-1] > ASK_PERIOD_S]:
            del asks[key]  # evict stale buckets so spoofed keys cannot grow memory
        for key, limit in ((ip, ASK_LIMIT), ("*", ASK_GLOBAL_LIMIT)):
            q = asks[key]
            while q and now - q[0] > ASK_PERIOD_S:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(429, f"max {limit} questions per {ASK_PERIOD_S // 60} min")
        asks[ip].append(now)
        asks["*"].append(now)

    class Q(BaseModel):
        q: str = Field(min_length=1, max_length=300)

    @api.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @api.get("/api/scans")
    def scans():
        fresh()  # aliases only, from ALLOWED_SCANS; never directory names
        return [a for a, (_, sid) in sources().items() if (DATA / sid / "index.json").exists()]

    @api.get("/api/scan/{alias}")
    def scan_doc(alias: str):
        doc = load(alias)
        return public_doc(doc, alias, [public(w) for w in doc["windows"]])

    @api.get("/api/scan/{alias}/search")
    def scan_search(alias: str, q: str):
        return search(load(alias), q[:300])

    @api.post("/api/scan/{alias}/ask")
    def scan_ask(alias: str, body: Q, request: Request):
        doc = load(alias)
        rate_limit(request)
        hits = search(doc, body.q, k=3)
        wins = sorted((w for w in doc["windows"] if w["id"] in {h["id"] for h in hits}), key=lambda w: w["start"])
        _, sid = source(alias)
        return {"answer": Reasoner().ask.remote(sid, body.q, wins), "windows": [public(w) for w in wins]}

    @api.post("/api/ingest/{alias}")
    def scan_ingest(alias: str, authorization: str = Header(default="")):
        # Ingestion pulls from storage with the server's token and spends H100 time: bearer-auth it.
        given = authorization.removeprefix("Bearer ").strip()
        if not given or not hmac.compare_digest(given, os.environ["INGEST_TOKEN"]):
            raise HTTPException(401, "bearer token required")
        source(alias)
        return {"call_id": ingest.spawn(alias).object_id}

    @api.get("/media/{alias}/{kind}/{name}")
    def media(alias: str, kind: str, name: str):
        if not (kind in ("clips", "thumbs") and NAME_RE.match(name)):
            raise HTTPException(400, "bad path")
        _, sid = source(alias)
        fresh()
        p = DATA / sid / kind / name
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="video/mp4" if kind == "clips" else "image/jpeg")

    return api


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Real-Time Video Agents</title>
<style>
:root{--bg:#0b0d10;--card:#14181d;--line:#232a32;--fg:#e8ecf1;--mute:#8b96a3;--acc:#76b900;--acc2:#3dd6f5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 16px 8px;max-width:1100px;margin:auto}h1{margin:0;font-size:20px;letter-spacing:.3px}
h1 b{color:var(--acc)}header p{margin:4px 0 0;color:var(--mute);font-size:13px}
main{max-width:1100px;margin:auto;padding:0 16px 48px}
.row{display:flex;gap:8px;margin:14px 0}input,select{flex:1;min-width:0;background:var(--card);border:1px solid var(--line);color:var(--fg);padding:11px 13px;border-radius:10px;font-size:15px}
button{background:var(--acc);color:#061000;border:0;padding:11px 16px;border-radius:10px;font-weight:600;cursor:pointer}button.alt{background:var(--acc2);color:#031218}button:disabled{opacity:.5}
.box{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}
.box h2{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.8px;color:var(--mute)}
#answer{white-space:pre-wrap;min-height:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;gap:0;flex-direction:column}
.card.hit{border-color:var(--acc)}.card .m{display:flex;gap:10px;padding:10px}
.card video,.card img{width:108px;height:192px;object-fit:cover;border-radius:8px;background:#000;flex:none}
.t{font-size:12px;color:var(--acc2);font-variant-numeric:tabular-nums}.s{font-size:11px;color:var(--mute);margin-left:6px}
.txt{font-size:13.5px;color:#cfd6de;margin:4px 0}.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.tag{font-size:11px;background:#1d242c;border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--mute)}.tag b{color:var(--fg)}
.pose{font-size:11px;color:var(--mute);margin-top:4px;font-variant-numeric:tabular-nums}
.dim{color:var(--mute);font-size:13px}
#rerun{display:none}#rerun iframe{width:100%;height:min(70vh,560px);border:0;border-radius:10px;background:#000}
details summary{cursor:pointer;color:var(--mute);font-size:13px}
</style></head><body>
<header><h1><b>Real-Time</b> Video Agents <span class=dim>· Cosmos Reason + YOLO11 + semantic search</span></h1>
<p>A Ray-Ban glasses walk-through, cut into windows, described for a robot and made searchable.</p></header>
<main>
<div class=row><select id=scan></select></div>
<div class=row><input id=q maxlength=300 placeholder="Search: where is the laptop? when did someone open the door?"><button id=sb>Search</button></div>
<div class=row><input id=aq maxlength=300 placeholder="Ask the agent anything about the scan"><button id=ab class=alt>Ask</button></div>
<div class=box id=abox style="display:none"><h2>Agent answer</h2><div id=answer></div></div>
<div class=box id=rerun><h2>Video in Rerun · 3D reconstruction of this scan</h2><details id=rrd><summary>Open the Rerun web viewer (loads the scene .rrd; a few seconds)</summary><iframe id=rr allow="fullscreen"></iframe></details></div>
<div class=box><h2 id=title>Timeline</h2><div class=grid id=grid></div></div>
</main>
<script>
const $=s=>document.querySelector(s);let doc=null,scan=null;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=s=>s.toFixed(1)+'s';
function card(w,hit){const id=String(w.id).padStart(3,'0');
const pose=w.pose?`<div class=pose>filmed at frame ${w.mid_frame+1} · camera at (${w.pose.map(v=>v.toFixed(1)).join(', ')})</div>`:'';
return `<div class="card${hit?' hit':''}"><div class=m>
<video src="/media/${esc(scan)}/clips/${id}.mp4" poster="/media/${esc(scan)}/thumbs/${id}.jpg" muted loop playsinline preload=none onmouseenter="this.play()" onmouseleave="this.pause()" onclick="this.paused?this.play():this.pause()"></video>
<div><span class=t>${fmt(w.start)} – ${fmt(w.end)}</span>${w.score!=null?`<span class=s>score ${esc(w.score)}</span>`:''}
<div class=txt>${esc(w.text)}</div><div class=tags>${w.tags.map(([c,n])=>`<span class=tag><b>${esc(c)}</b> ×${esc(n)}</span>`).join('')}</div>${pose}</div></div></div>`}
function render(ws,title,hit){$('#title').textContent=title;$('#grid').innerHTML=ws.map(w=>card(w,hit)).join('')||'<span class=dim>No results.</span>'}
async function loadScan(id){scan=id;doc=await (await fetch(`/api/scan/${encodeURIComponent(id)}`)).json();
const rr=$('#rerun');$('#rrd').open=false;$('#rr').src='about:blank';
if(doc.rerun_url&&doc.rerun_url.startsWith('https://app.rerun.io/')){rr.style.display='block';$('#rrd').ontoggle=()=>{if($('#rrd').open&&$('#rr').src!==doc.rerun_url)$('#rr').src=doc.rerun_url}}else{rr.style.display='none'}
render(doc.windows,`Timeline · ${doc.windows.length} windows · ${doc.duration.toFixed(0)} s · ${doc.n_frames} frames${doc.place?' · '+doc.place:''}${doc.rendered?' · rendered from the 3D splat':''} · ${doc.model.split('/')[1]}`)}
async function search(){const q=$('#q').value.trim();if(!q)return loadScan(scan);
const r=await (await fetch(`/api/scan/${encodeURIComponent(scan)}/search?q=${encodeURIComponent(q)}`)).json();render(r,`Results for “${q}”`,true)}
async function ask(){const q=$('#aq').value.trim();if(!q)return;$('#ab').disabled=true;$('#abox').style.display='';$('#answer').textContent='Thinking… (first question wakes the H100, about a minute)';
try{const res=await fetch(`/api/scan/${encodeURIComponent(scan)}/ask`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({q})});const r=await res.json();
if(!res.ok)throw new Error(r.detail||res.status);$('#answer').textContent=r.answer;render(r.windows,`Windows the agent looked at for “${q}”`,true)}catch(e){$('#answer').textContent='Error: '+e.message}$('#ab').disabled=false}
$('#sb').onclick=search;$('#q').onkeydown=e=>e.key=='Enter'&&search();$('#ab').onclick=ask;$('#aq').onkeydown=e=>e.key=='Enter'&&ask();
$('#scan').onchange=e=>loadScan(e.target.value);
(async()=>{const ids=await (await fetch('/api/scans')).json();$('#scan').innerHTML=ids.map(i=>`<option>${esc(i)}</option>`).join('');
const want=new URLSearchParams(location.search).get('scan');if(ids.length)loadScan(ids.includes(want)?want:ids[0]);else $('#title').textContent='No scans indexed yet'})();
</script></body></html>"""
