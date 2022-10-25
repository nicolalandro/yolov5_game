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


class VideoProcessor:
    points = 0
    enemy = (256, 256)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        # return av.VideoFrame.from_ndarray(np.array(im_pil, copy=False), format="bgr24")
        results = st.model(im_pil, size=112)
        bbox_img = np.array(results.render()[0])

        # il draw è molto lento
        res_im_pil = Image.fromarray(bbox_img)
        draw = ImageDraw.Draw(res_im_pil)
        draw.text((28, 36), "Points:" +
                  str(VideoProcessor.points), fill=(255, 0, 0))

        if VideoProcessor.points >= 1:
            draw.text((res_im_pil.width//2, res_im_pil.height//2),
                      "You WIN", fill=(255, 0, 0))
        else:
            draw.text(VideoProcessor.enemy, "ENEMY", fill=(255, 0, 0))

            for p in results.pred[0]:
                x1, y1, x2, y2, conf, pred = list(p.numpy())
                class_name = results.names[int(pred)]
                if VideoProcessor.enemy[0] > x1 and VideoProcessor.enemy[0] < x2 and \
                        VideoProcessor.enemy[1] > y1 and VideoProcessor.enemy[1] < y2:
                    VideoProcessor.points += 1

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
