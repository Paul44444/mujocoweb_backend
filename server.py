from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import json
import os
from pathlib import Path
from editor_api import SceneAsset, _authorize as authorize_editor, router as editor_router
import queue
import re
import threading
import time
import traceback
from typing import Any, Dict

# These must be set before importing MuJoCo, RoboHive, or muj1.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from object_generator import generate_object, validate_object_spec


def _integer_setting(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        print(
            f"Ignoring invalid {name}={raw_value!r}; using {default}",
            flush=True,
        )
        return default
    if not minimum <= value <= maximum:
        print(
            f"Ignoring out-of-range {name}={value}; using {default}",
            flush=True,
        )
        return default
    return value


JPEG_QUALITY = _integer_setting("JPEG_QUALITY", 70, 1, 100)
STREAM_FPS = _integer_setting("STREAM_FPS", 30, 1, 60)
PERFORMANCE_LOG_INTERVAL = _integer_setting(
    "PERFORMANCE_LOG_INTERVAL",
    100,
    1,
    10_000,
)

app = FastAPI(title="MuJoCo + Isaac Lab Web Backend")
app.include_router(editor_router)
PUBLIC_SCENE_PATH = re.compile(r"^/api/editor/users(?:/[^/]+/scenes(?:/[^/]+)?)?$")


@app.middleware("http")
async def guard_editor_requests(request: Request, call_next):
    if request.url.path.startswith("/api/editor") and request.method != "OPTIONS":
        if not PUBLIC_SCENE_PATH.fullmatch(request.url.path):
            try:
                authorize_editor(request.headers.get("authorization"))
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        if request.method in {"PUT", "POST"}:
            try:
                if int(request.headers.get("content-length", "0")) > 170_000:
                    return JSONResponse({"detail": "Editor request is too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
    return await call_next(request)
generation_requests = defaultdict(deque)
generation_lock = threading.Lock()
GENERATION_LIMIT = 12
GENERATION_WINDOW_SECONDS = 60 * 60

AVAILABLE_TASKS = {"relocate", "hammer", "door", "pen"}
DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://mujocoweb.vercel.app",
)
ISAAC_OUTPUT_DIRECTORY = os.environ.get("ISAAC_OUTPUT_DIRECTORY", "/tmp/mujocoweb-isaac")
ISAAC_FRAME_PATH = os.path.join(ISAAC_OUTPUT_DIRECTORY, "frame.jpg")
ISAAC_METADATA_PATH = os.path.join(ISAAC_OUTPUT_DIRECTORY, "metadata.json")
ISAAC_STATUS_PATH = os.path.join(ISAAC_OUTPUT_DIRECTORY, "status.json")
frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "FRONTEND_ORIGINS",
        ",".join(DEFAULT_FRONTEND_ORIGINS),
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "MuJoCo and Isaac Lab backend gateway is running"}


async def stream_isaac_simulation(websocket: WebSocket) -> None:
    """Relay frames from the persistent Isaac Lab process to one browser."""
    await websocket.send_json(
        {
            "type": "status",
            "status": "simulation_started",
            "engine": "isaaclab",
            "task": "Isaac-Lift-Cube-Franka-v0",
        }
    )
    last_frame_mtime = 0
    waiting_since = time.monotonic()
    loop = asyncio.get_running_loop()
    paused = asyncio.Event()
    disconnected = asyncio.Event()

    async def receive_commands() -> None:
        while True:
            try:
                command = await websocket.receive_json()
            except WebSocketDisconnect:
                disconnected.set()
                return
            except asyncio.CancelledError:
                return
            except Exception:
                disconnected.set()
                return
            if command.get("type") == "set_paused":
                if command.get("paused") is True:
                    paused.set()
                elif command.get("paused") is False:
                    paused.clear()

    receiver_task = asyncio.create_task(receive_commands())

    try:
        while not disconnected.is_set():
            if paused.is_set():
                await asyncio.sleep(0.05)
                continue
            try:
                frame_stat = await loop.run_in_executor(None, os.stat, ISAAC_FRAME_PATH)
            except FileNotFoundError:
                if time.monotonic() - waiting_since > 180:
                    status = "Isaac Lab worker did not become ready."
                    try:
                        with open(ISAAC_STATUS_PATH, encoding="utf-8") as status_file:
                            status_data = json.load(status_file)
                        status = f"Isaac Lab worker status: {status_data.get('status', 'unknown')}"
                    except (OSError, ValueError):
                        pass
                    await websocket.send_json({"type": "error", "message": status})
                    return
                await asyncio.sleep(0.25)
                continue

            if frame_stat.st_mtime_ns == last_frame_mtime:
                await asyncio.sleep(0.025)
                continue

            last_frame_mtime = frame_stat.st_mtime_ns
            jpeg_bytes = await loop.run_in_executor(None, Path(ISAAC_FRAME_PATH).read_bytes)
            metadata: dict[str, Any] = {}
            try:
                metadata_bytes = await loop.run_in_executor(None, Path(ISAAC_METADATA_PATH).read_bytes)
                metadata = json.loads(metadata_bytes)
            except (OSError, ValueError, TypeError):
                pass
            await websocket.send_text(json.dumps({"type": "frame_metadata", **metadata}))
            await websocket.send_bytes(jpeg_bytes)
    except WebSocketDisconnect:
        print("Browser disconnected from Isaac Lab stream", flush=True)
    except Exception:
        print("Isaac Lab WebSocket relay crashed:", flush=True)
        traceback.print_exc()
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass


class ObjectPrompt(BaseModel):
    description: str


@app.post("/api/objects/generate")
def generate_simulation_object(prompt: ObjectPrompt, request: Request) -> Dict[str, Any]:
    client_id = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if not client_id:
        client_id = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with generation_lock:
        requests = generation_requests[client_id]
        while requests and requests[0] <= now - GENERATION_WINDOW_SECONDS:
            requests.popleft()
        if len(requests) >= GENERATION_LIMIT:
            raise HTTPException(status_code=429, detail="Generation limit reached. Please try again later.")
        requests.append(now)
    try:
        return generate_object(prompt.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        print(f"Object generation error: {exc}", flush=True)
        raise HTTPException(status_code=502, detail="The AI could not create an object right now. Please try again.") from exc


@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket) -> None:
    print("WebSocket connection attempt received", flush=True)

    await websocket.accept()

    engine = websocket.query_params.get("engine", "mujoco").lower()
    if engine == "isaaclab":
        print("Browser connected to persistent Isaac Lab stream", flush=True)
        await stream_isaac_simulation(websocket)
        return
    if engine != "mujoco":
        await websocket.send_json({"type": "error", "message": f"Unknown simulation engine: {engine}"})
        await websocket.close(code=1008)
        return

    task_id = websocket.query_params.get("task", "relocate").lower()
    editor_mode = websocket.query_params.get("editor") == "1"
    scene_assets = []
    editor_camera = None
    raw_camera = websocket.query_params.get("camera")
    if editor_mode and raw_camera:
        try:
            if len(raw_camera) > 500:
                raise ValueError("Camera data is too long")
            camera_data = json.loads(raw_camera)
            if not isinstance(camera_data, dict):
                raise ValueError("Camera must be an object")
            azimuth = float(camera_data["azimuth"])
            elevation = float(camera_data["elevation"])
            distance = float(camera_data["distance"])
            lookat = [float(value) for value in camera_data["lookat"]]
            if (
                not all(np.isfinite(value) for value in [azimuth, elevation, distance, *lookat])
                or len(lookat) != 3
                or not -85 <= elevation <= -5
                or not 0.45 <= distance <= 5
                or any(abs(value) > 10 for value in lookat)
            ):
                raise ValueError("Camera values are out of range")
            editor_camera = {"azimuth": azimuth, "elevation": elevation, "distance": distance, "lookat": lookat}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid camera: {exc}"})
            await websocket.close(code=1008)
            return
    raw_scene = websocket.query_params.get("scene")
    if raw_scene:
        if task_id != "relocate" or len(raw_scene) > 12_000:
            await websocket.send_json({"type": "error", "message": "Custom scenes currently support Relocate only."})
            await websocket.close(code=1008)
            return
        try:
            scene_data = json.loads(raw_scene)
            if not isinstance(scene_data, list) or len(scene_data) > 25:
                raise ValueError("Scene must contain at most 25 assets")
            scene_assets = [SceneAsset.model_validate(item).model_dump() for item in scene_data]
        except (ValueError, TypeError) as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid scene: {exc}"})
            await websocket.close(code=1008)
            return
    if task_id not in AVAILABLE_TASKS:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Unknown simulation task: {task_id}",
            }
        )
        await websocket.close(code=1008)
        return

    object_spec = None
    raw_object_spec = websocket.query_params.get("object")
    if raw_object_spec:
        if task_id != "relocate" or len(raw_object_spec) > 8_000:
            await websocket.send_json({"type": "error", "message": "Custom objects are currently supported only for Relocate."})
            await websocket.close(code=1008)
            return
        try:
            object_spec = validate_object_spec(json.loads(raw_object_spec))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid custom object: {exc}"})
            await websocket.close(code=1008)
            return

    print("Browser connected to simulation WebSocket", flush=True)
    print(
        "Rendering configuration:",
        {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        flush=True,
    )

    # Keep only the newest rendered frame.
    frame_queue: queue.Queue[
        tuple[bytes, dict[str, Any]]
    ] = queue.Queue(maxsize=1)

    # Keep only the newest browser click.
    target_queue: queue.Queue[
        dict[str, float]
    ] = queue.Queue(maxsize=1)
    control_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
    pause_event = threading.Event()
    stop_event = threading.Event()

    simulation_finished = threading.Event()
    simulation_error: list[str] = []

    frame_count = 0
    jpeg_encode_seconds = 0.0
    callback_started = time.perf_counter()

    def frame_callback(frame: Any, metadata: Any) -> None:
        """
        Called from the MuJoCo simulation thread.

        Converts an RGB NumPy frame into JPEG bytes and stores only
        the newest frame in frame_queue.
        """
        nonlocal frame_count, jpeg_encode_seconds

        try:
            if frame is None:
                raise ValueError("frame_callback received frame=None")

            frame_array = np.asarray(frame)

            if frame_array.ndim != 3:
                raise ValueError(
                    "Expected a 3-dimensional image array, "
                    f"but received shape {frame_array.shape}"
                )

            if frame_array.shape[2] not in (3, 4):
                raise ValueError(
                    "Expected an RGB or RGBA image, "
                    f"but received shape {frame_array.shape}"
                )

            if frame_array.dtype != np.uint8:
                # Some renderers return floating-point RGB values in [0, 1].
                if np.issubdtype(frame_array.dtype, np.floating):
                    frame_array = np.clip(
                        frame_array * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    frame_array = frame_array.astype(np.uint8)

            if frame_array.shape[2] == 4:
                frame_bgr = cv2.cvtColor(
                    frame_array,
                    cv2.COLOR_RGBA2BGR,
                )
            else:
                frame_bgr = cv2.cvtColor(
                    frame_array,
                    cv2.COLOR_RGB2BGR,
                )

            encode_started = time.perf_counter()
            success, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            jpeg_encode_seconds += time.perf_counter() - encode_started

            if not success:
                raise RuntimeError("OpenCV failed to encode JPEG frame")

            if not isinstance(metadata, dict):
                metadata = {}

            frame_count += 1

            if frame_count == 1:
                print(
                    "First frame callback received:",
                    {
                        "shape": frame_array.shape,
                        "dtype": str(frame_array.dtype),
                        "jpeg_bytes": len(encoded),
                        "metadata": metadata,
                    },
                    flush=True,
                )
            elif frame_count % PERFORMANCE_LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - callback_started
                print(
                    "Frame processing performance:",
                    f"produced_fps={frame_count / elapsed:.2f},",
                    f"jpeg_avg_ms="
                    f"{jpeg_encode_seconds * 1000 / frame_count:.2f},",
                    f"jpeg_quality={JPEG_QUALITY}",
                    flush=True,
                )

            item = (
                encoded.tobytes(),
                metadata,
            )

            # Discard the previous frame if the browser is slower
            # than the simulation.
            try:
                frame_queue.put_nowait(item)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    frame_queue.put_nowait(item)
                except queue.Full:
                    pass

        except Exception:
            print(
                "Exception inside frame_callback:",
                flush=True,
            )
            traceback.print_exc()

            if not simulation_error:
                simulation_error.append(
                    "Frame processing failed. "
                    "See the Render log for the full traceback."
                )

            simulation_finished.set()
            raise

    async def receive_browser_commands() -> None:
        """
        Receive normalized target coordinates from the browser.
        """
        while True:
            try:
                data = await websocket.receive_json()

            except WebSocketDisconnect:
                print(
                    "Browser disconnected while receiving commands",
                    flush=True,
                )
                return

            except asyncio.CancelledError:
                return

            except Exception:
                print(
                    "Could not receive browser command:",
                    flush=True,
                )
                traceback.print_exc()
                return

            command_type = data.get("type")

            if command_type == "scene_transform" and editor_mode:
                try:
                    index = data["index"]
                    position = data["position"]
                    rotation = data["rotation"]
                    if type(index) is not int or not 0 <= index < len(scene_assets):
                        raise ValueError("Invalid asset index")
                    if not isinstance(position, list) or not isinstance(rotation, list) or len(position) != 3 or len(rotation) != 3:
                        raise ValueError("Invalid transform")
                    position = [float(value) for value in position]
                    rotation = [float(value) for value in rotation]
                    if not all(np.isfinite(value) for value in position + rotation):
                        raise ValueError("Non-finite transform")
                    if any(abs(value) > 5 for value in position) or any(abs(value) > 3600 for value in rotation):
                        raise ValueError("Transform out of range")
                    control_queue.put_nowait({"type": "scene_transform", "index": index, "position": position, "rotation": rotation})
                except (KeyError, TypeError, ValueError, OverflowError, queue.Full):
                    pass
                continue

            if command_type == "set_paused":
                if data.get("paused") is True:
                    pause_event.set()
                elif data.get("paused") is False:
                    pause_event.clear()
                continue

            if command_type == "camera_reset":
                try:
                    control_queue.put_nowait({"type": "camera_reset"})
                except queue.Full:
                    pass
                continue

            if command_type in {"camera_orbit", "camera_zoom"}:
                try:
                    if command_type == "camera_orbit":
                        command = {
                            "type": command_type,
                            "delta_x": max(-100.0, min(100.0, float(data["deltaX"]))),
                            "delta_y": max(-100.0, min(100.0, float(data["deltaY"]))),
                        }
                    else:
                        command = {
                            "type": command_type,
                            "delta": max(-1.0, min(1.0, float(data["delta"]))),
                        }
                    control_queue.put_nowait(command)
                except (KeyError, TypeError, ValueError, queue.Full):
                    pass
                continue

            if command_type != "set_target":
                print(
                    "Ignoring unknown browser command:",
                    data,
                    flush=True,
                )
                continue

            try:
                u = float(data["u"])
                v = float(data["v"])
            except (KeyError, TypeError, ValueError):
                print(
                    "Invalid target command:",
                    data,
                    flush=True,
                )
                continue

            if not (
                0.0 <= u <= 1.0
                and 0.0 <= v <= 1.0
            ):
                print(
                    "Target coordinates outside image:",
                    u,
                    v,
                    flush=True,
                )
                continue

            target = {
                "u": u,
                "v": v,
            }

            # Discard an older, unprocessed click.
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                target_queue.put_nowait(target)
            except queue.Full:
                pass

            print(
                "Received browser target:",
                f"u={u:.4f},",
                f"v={v:.4f}",
                flush=True,
            )

    def simulation_worker() -> None:
        """
        Import and run the heavy MuJoCo/RoboHive code only after
        the browser has opened the WebSocket connection.
    
        This allows Uvicorn to start quickly and bind Render's port
        before RoboHive and Torch are imported.
        """
        print("Simulation worker entered", flush=True)
    
        try:
            
            import time
            
            print(
                "Importing MuJoCo simulation code...",
                flush=True,
            )
            
            import_started = time.perf_counter()
            
            from muj1 import run_simulation
            
            import_duration = time.perf_counter() - import_started
            
            print(
                f"MuJoCo simulation code imported in "
                f"{import_duration:.2f} seconds",
                flush=True,
            )
            
            print(
                "Calling run_simulation()",
                flush=True,
            )
    
            run_simulation(
                frame_callback=frame_callback,
                target_queue=target_queue,
                control_queue=control_queue,
                pause_event=pause_event,
                task_id=task_id,
                object_spec=object_spec,
                editor_mode=editor_mode,
                editor_camera=editor_camera,
                scene_assets=scene_assets,
                stop_event=stop_event,
            )
    
            print(
                "run_simulation() returned normally",
                flush=True,
            )
    
        except Exception as exc:
            error_message = (
                f"{type(exc).__name__}: {exc}"
            )
    
            simulation_error.append(error_message)
    
            print(
                "Simulation worker crashed:",
                error_message,
                flush=True,
            )
            traceback.print_exc()
    
        finally:
            simulation_finished.set()
    
            print(
                "Simulation worker finished",
                flush=True,
            )
    
    simulation_thread = threading.Thread(
        target=simulation_worker,
        name="mujoco-simulation-thread",
        daemon=True,
    )

    print("Starting simulation thread", flush=True)
    simulation_thread.start()

    receiver_task = asyncio.create_task(
        receive_browser_commands()
    )

    sent_frame_count = 0
    send_started = time.perf_counter()
    last_frame_sent_at = 0.0
    empty_queue_count = 0

    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": "simulation_started",
                "task": task_id,
            }
        )

        print(
            "Sent simulation_started status to browser",
            flush=True,
        )

        while True:
            try:
                loop = asyncio.get_running_loop()

                jpeg_bytes, metadata = await loop.run_in_executor(
                    None,
                    frame_queue.get,
                    True,
                    0.5,
                )

                empty_queue_count = 0

            except queue.Empty:
                empty_queue_count += 1

                if simulation_finished.is_set():
                    print(
                        "Simulation finished while frame queue was empty",
                        flush=True,
                    )
                    break

                # Print a diagnostic approximately every five seconds.
                if empty_queue_count % 10 == 0:
                    print(
                        "Still waiting for first/new frame. "
                        f"Worker alive={simulation_thread.is_alive()}, "
                        f"frames produced={frame_count}",
                        flush=True,
                    )

                continue

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "frame_metadata",
                        **metadata,
                    }
                )
            )

            await websocket.send_bytes(jpeg_bytes)

            sent_frame_count += 1
            elapsed_since_last_frame = time.perf_counter() - last_frame_sent_at
            minimum_frame_interval = 1.0 / STREAM_FPS
            if elapsed_since_last_frame < minimum_frame_interval:
                await asyncio.sleep(minimum_frame_interval - elapsed_since_last_frame)
            last_frame_sent_at = time.perf_counter()

            if sent_frame_count == 1:
                print(
                    "First JPEG frame sent to browser:",
                    f"{len(jpeg_bytes)} bytes",
                    flush=True,
                )
            elif sent_frame_count % PERFORMANCE_LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - send_started
                print(
                    "WebSocket performance:",
                    f"sent_fps={sent_frame_count / elapsed:.2f},",
                    f"sent_frames={sent_frame_count}",
                    flush=True,
                )

        if simulation_error:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": simulation_error[0],
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "status",
                    "status": "simulation_finished",
                }
            )

    except WebSocketDisconnect:
        print(
            "Browser disconnected from simulation",
            flush=True,
        )

    except Exception:
        print(
            "WebSocket handler crashed:",
            flush=True,
        )
        traceback.print_exc()

    finally:
        stop_event.set()
        pause_event.clear()
        receiver_task.cancel()

        try:
            await receiver_task
        except asyncio.CancelledError:
            pass

        print(
            "Simulation WebSocket closed",
            flush=True,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
