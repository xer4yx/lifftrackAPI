from queue import Queue

import cv2
import asyncio

frame_queue = Queue(maxsize=30)
result_queue = Queue(maxsize=30)


async def generate_frames():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not frame_queue.full():
                frame_queue.put(frame)

            if not result_queue.empty():
                annotated_frame = result_queue.get()
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            await asyncio.sleep(0.01)
    finally:
        cap.release()
