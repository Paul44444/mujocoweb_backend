import asyncio
import json
import os
import queue
import threading
import traceback
from typing import Any

# These must be set before importing MuJoCo, RoboHive, or muj1.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MuJoCo Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # Add your Vercel frontend URL here later, for example:
        # "https://your-project.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "MuJoCo backend is running"}


@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket) -> None:
    print("WebSocket connection attempt received", flush=True)

    await websocket.accept()

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

    simulation_finished = threading.Event()
    simulation_error: list[str] = []

    frame_count = 0

    def frame_callback(frame: Any, metadata: Any) -> None:
        """
        Called from the MuJoCo simulation thread.

        Converts an RGB NumPy frame into JPEG bytes and stores only
        the newest frame in frame_queue.
        """
        nonlocal frame_count

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

            success, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

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
            elif frame_count % 100 == 0:
                print(
                    f"Processed {frame_count} rendered frames",
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

            if data.get("type") != "set_target":
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
            print(
                "Importing MuJoCo simulation code...",
                flush=True,
            )
    
            # Deliberately import here rather than at the top of server.py.
            from muj1 import run_simulation
    
            print(
                "MuJoCo simulation code imported",
                flush=True,
            )
            print(
                "Calling run_simulation()",
                flush=True,
            )
    
            run_simulation(
                frame_callback=frame_callback,
                target_queue=target_queue,
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
    empty_queue_count = 0

    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": "simulation_started",
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

            if sent_frame_count == 1:
                print(
                    "First JPEG frame sent to browser:",
                    f"{len(jpeg_bytes)} bytes",
                    flush=True,
                )
            elif sent_frame_count % 100 == 0:
                print(
                    f"Sent {sent_frame_count} frames to browser",
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