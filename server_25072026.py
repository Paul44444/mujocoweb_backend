import asyncio
import json
import queue
import threading
from typing import Any

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from muj1 import run_simulation
from typing import Dict

app = FastAPI(title="MuJoCo Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "MuJoCo backend is running"}


@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    # Maximum size 1 prevents latency from accumulating.
    # If the browser is slower than MuJoCo, old frames are discarded.
    frame_queue: queue.Queue[tuple[bytes, dict[str, Any]]] = queue.Queue(
        maxsize=1
    )

    simulation_finished = threading.Event()
    simulation_error: list[str] = []

    # Browser click commands are passed from the WebSocket task
    # into the separate MuJoCo simulation thread.
    target_queue: queue.Queue[dict[str, float]] = queue.Queue(
        maxsize=1
    )

    def frame_callback(frame, metadata) -> None:
        """Called synchronously from the MuJoCo simulation thread."""

        print(
            "FRAME RECEIVED",
            frame.shape,
            metadata.get("step"),
        )
        
        # MuJoCo provides RGB; OpenCV JPEG encoding expects BGR.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        success, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if not success:
            return

        item = encoded.tobytes(), metadata

        # Drop the old frame when the queue is full.
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
    
async def receive_browser_commands() -> None:
    """
    Receive JSON commands from the browser while the simulation
    continues running in its separate thread.
    """

    while True:
        
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            print("Browser disconnected from simulation")

        except Exception as exc:
            print(f"WebSocket error: {exc}")

        finally:
            receiver_task.cancel()

            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
        
        if data.get("type") != "set_target":
            print("Ignoring unknown browser command:", data)
            continue

        try:
            u = float(data["u"])
            v = float(data["v"])
        except (KeyError, TypeError, ValueError):
            print("Invalid target command:", data)
            continue

        if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
            print("Target coordinates outside image:", u, v)
            continue

        target = {
            "u": u,
            "v": v,
        }

        # Remove an older click if it has not yet been processed.
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            target_queue.put_nowait(target)
        except queue.Full:
            pass

        print(
            f"Received browser target: "
            f"u={u:.4f}, v={v:.4f}"
        )
    
    def simulation_worker():
        print("Simulation worker started")
        
        try:
            run_simulation(
                frame_callback=frame_callback,
                target_queue=target_queue,
            )
            print("Simulation worker finished")
        except Exception as exc:
            simulation_error.append(str(exc))
            print("Simulation error:", repr(exc))
        finally:
            simulation_finished.set()

    simulation_thread = threading.Thread(
        target=simulation_worker,
        daemon=True,
    )
    simulation_thread.start()

    receiver_task = asyncio.create_task(
        receive_browser_commands()
    )

    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": "simulation_started",
            }
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
            except queue.Empty:
                if simulation_finished.is_set():
                    break
                continue

            # First send metadata as text.
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "frame_metadata",
                        **metadata,
                    }
                )
            )

            # Then send the corresponding JPEG as binary data.
            await websocket.send_bytes(jpeg_bytes)

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
        print("Browser disconnected from simulation")

    except Exception as exc:
        print(f"WebSocket error: {exc}")

    finally:
        receiver_task.cancel()
    try:
        await receiver_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )