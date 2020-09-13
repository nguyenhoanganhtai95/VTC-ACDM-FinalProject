# coding: utf-8
import cv2
import os
import numpy as np
import time
from termcolor import colored
from helper import read_pkl_model, start_up_init, encode_image
from multiprocessing.dummy import Process, Queue
import asyncio
import socketio
import face_embedding
import face_detector


async def upload_loop(url="http://127.0.0.1:6789"):
    # =====================Uploader Setsup========================
    sio = socketio.AsyncClient()

    @sio.on("response", namespace="/remilia")
    async def on_response(data):
        image_string = 0
        result_string = 0
        for pair in upstream_queue.get():
            if pair[0] == data:
                image_string = encode_image(pair[-1])
                if len(pair) is 3:
                    result_string = pair[1]
                break

        jstr = {"frame": image_string, "result": result_string}
        await sio.emit("frame_data", jstr, namespace="/remilia")

        try:
            img, dt, prob, name, ip = result_queue.get_nowait()
            jstr = {
                "image": encode_image(img),
                "time": dt,
                "name": name,
                "prob": prob,
                "ip": ip,
            }
            await sio.emit("result_data", jstr, namespace="/remilia")
        except:
            pass

    @sio.on("connect", namespace="/remilia")
    async def on_connect():
        jstr = {"frame": 0, "result": 0}
        await sio.emit("frame_data", jstr, namespace="/remilia")

    await sio.connect(url)
    await sio.wait()


async def embedding_loop(preload):
    # =================== FR MODEL ====================
    embedding = face_embedding.EmbeddingModel(preload)
    while True:
        result = embedding.arcface_deal(suspicion_face_queue.get())
        result_queue.put(result)
        # print(colored(f"Embedding cost: {loop.time() - start_time}", "red"), flush=True)
    # [[0.30044544 0.31831665 0.30363247 0.07760544]]


async def detection_loop(preload):
    # =================== FD MODEL ====================
    detector = face_detector.DetectorModel(preload)
    embedding_threshold = preload.embedding_threshold
    rate = preload.max_frame_rate
    loop = asyncio.get_running_loop()

    def retina_deal(pair):
        address, frame = pair
        res = []
        for img, box in detector.get_all_boxes(frame, save_img=False):
            res.append(box.tolist())
            if box[4] > embedding_threshold:
                try:
                    suspicion_face_queue.put_nowait((address, img))
                except:
                    pass

        return (address, frame) if res == [] else (address, res, frame)

    while True:
        start_time = loop.time()
        frame_list = [retina_deal(pair) for pair in frame_queue.get()]
        print(colored(f'Detection cost: {loop.time() - start_time}', 'red'),
              flush=True)
        upstream_queue.put(frame_list)
        for _ in range(int((loop.time() - start_time) * rate)):
            upstream_queue.put(frame_queue.get())


async def camera_loop(preload):
    rate = 1 / preload.max_frame_rate
    code_list = preload.usb_camera_code

    # =================== ETERNAL LOOP ====================
    loop = asyncio.get_running_loop()
    while True:
        start_time = loop.time()
        camera.read()[1]
        # pairs = [(ip, camera_dict[ip].frame(rows=672, cols=672)) for ip in addr_list]
        # pairs = [(str(code), camera_dict[code].read()[1]) for code in code_list]
        pairs = [(str(code), cv2.resize(camera_dict[code].read()[1], (672, 672)))
                 for code in code_list]
        frame_queue.put(pairs)
        #print(loop.time() - start_time, frame_queue.qsize())
        restime = rate - loop.time() + start_time
        if restime > 0:
            await asyncio.sleep(restime)


# =================== INIT ====================
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
preload = start_up_init()
# preload.scales = [0.7]
# preload.usb_camera_code = [0]
preload.max_face_number = 30

print(preload.usb_camera_code)

camera_dict = {}
for code in preload.usb_camera_code:
    camera = cv2.VideoCapture(code)
    camera = cv2.VideoCapture('./Media/VideoTest.mp4')
    camera_dict[code] = camera

# =================== QUEUE ====================
frame_buffer_size = preload.queue_buffer_size
frame_queue = Queue(frame_buffer_size)
upstream_queue = Queue(frame_buffer_size)
suspicion_face_queue = Queue(preload.max_face_number)
result_queue = Queue(preload.max_face_number)

# =================== Process On ====================
Process(target=lambda: asyncio.run(embedding_loop(preload))).start()
Process(target=lambda: asyncio.run(detection_loop(preload))).start()
Process(target=lambda: asyncio.run(upload_loop())).start()
asyncio.run(camera_loop(preload))
