import streamlit as st
from PIL import Image, ImageDraw
import torch
import numpy as np

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av


# trick to load model once at server start
if not hasattr(st, 'classifier'):
    st.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  _verbose=False)
    # if you train your custm model "yolov5s.pt" use the following line
    # st.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', _verbose=False)


# Configuration for use camera from web deployed server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def hit(player, enemy, ray):
    xp, yp = player
    xe, ye = enemy
    dist = np.sqrt((xp - xe)**2 + (yp - ye)**2)
    return dist < ray


def compute_shape(arr):
    cx, cy = arr
    return [cx-10, cy-10, cx+10, cy+10]

class VideoProcessor:
    points = 0
    enemy = (256, 256)
    enemy_ray = 20
    player_pos = [50, 50]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        # return av.VideoFrame.from_ndarray(np.array(im_pil, copy=False), format="bgr24")
        results = st.model(im_pil, size=226)
        # bbox_img = np.array(results.render()[0])
        # res_im_pil = Image.fromarray(bbox_img)
        res_im_pil = im_pil

        # il draw è molto lento
        draw = ImageDraw.Draw(im_pil)
        draw.text((28, 36), "Points:" +
                  str(VideoProcessor.points), fill=(255, 0, 0))

        if VideoProcessor.points >= 10:
            draw.text((res_im_pil.width//2, res_im_pil.height//2),
                      "You WIN", fill=(255, 0, 0))
        else:
            draw.text(VideoProcessor.enemy, "ENEMY", fill=(255, 0, 0))

            for p in results.pred[0]:
                x1, y1, x2, y2, conf, pred = list(p.numpy())
                cx,cy = np.abs(x1-x2) / 2 + min([x1, x2]), np.abs(y1-y2) / 2 + min([y1, y2]) 
                class_name = results.names[int(pred)]
                if class_name == 'scissors':
                    VideoProcessor.player_pos = [cx, cy]
                    break
            
            shape = compute_shape(VideoProcessor.player_pos)
            draw.rectangle(shape, fill ="#ffff33", outline ="red")
                    
            if hit(VideoProcessor.player_pos, VideoProcessor.enemy, VideoProcessor.enemy_ray):
                VideoProcessor.points += 1
                VideoProcessor.enemy = list(np.random.randint(512 - 40 , size=2) +20 ) 
            # if VideoProcessor.enemy[0] > x1 and VideoProcessor.enemy[0] < x2 and \
                    #         VideoProcessor.enemy[1] > y1 and VideoProcessor.enemy[1] < y2:
                    #     VideoProcessor.points += 1

        # res_im_pil.save('test.png')
        return av.VideoFrame.from_ndarray(np.array(res_im_pil, copy=False), format="bgr24")


if __name__ == '__main__':
    st.session_state['game'] = {
        'points': 0
    }

    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )
