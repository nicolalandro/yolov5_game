[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nicolalandro-yolov5-game-streamlit-app-tjqz93.streamlitapp.com/)

# yolov5_game
A game with people detection.

## Develop
```
python3.8 -m venv venv
source venv/bin/activate
# or on fish shell
source venv/bin/activate.fish

# on linux: apt install python3-opencv
pip install -r requirements.txt

python -m streamlit run streamlit_app.py
```

# References
* [streamlit](https://streamlit.io/): library for web demo
* [streamlit_webrtc](https://github.com/whitphx/streamlit-webrtc): to use webcam from browser on stramlit
* [yolov5](https://github.com/ultralytics/yolov5): trained model
* [streamlit.io](https://streamlit.io/): to deploy web demo
