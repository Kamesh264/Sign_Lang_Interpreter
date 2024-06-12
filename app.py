# import streamlit as st
# from streamlit_webrtc import webrtc_streamer,  RTCConfiguration, VideoProcessorBase, WebRtcMode # type: ignore

# import av  # type: ignore
# import numpy as np
# import cv2
# import pickle
# import mediapipe as mp
# import copy
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import logging

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# # import os
# # os.environ["TF_FORCE_CPU"] = 'true'

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# confidence = 0.5
# # hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=confidence)
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=confidence)


# with open('trained_models/scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)
# with open('trained_models/random_forest_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)


# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

# st.set_page_config(page_title="ISLT", page_icon="🤖")

# st.title("Real Time Hand Gesture recognition and Live Hand Sign Translator")
# st.text("Developed by Batch - 17")

# df = pd.read_csv("landmark_data/Gestures_sentences.csv")
# my_dict = df.set_index('gesture_names')['sentence'].to_dict()
# final_dict = {}
# for key in my_dict:
#     t = []
#     words = key.split(',')
#     for word in words:
#         t.append(word)
#     s = ' '.join(t)
#     final_dict[s] = my_dict[key]


# word_limit = 3 
# def generate_caption(word, seq):
#     res = ''
#     if len(seq) < word_limit:
#         seq.append(word)
#         seq.append(word)
#         s = ' '.join(seq)
#         if s in final_dict:
#             res = final_dict[s]

#     elif len(seq) == word_limit:
#         seq.pop(0)
#         s = ' '.join(seq)
#         if s in final_dict:
#             res = final_dict[s]
#         seq.append(word)
#         s = ' '.join(seq)
#         if s in final_dict:
#             res = final_dict[s]   
#     return res


# class VideoProcessor(VideoProcessorBase):
#     threshold_list = []
#     threshold = 20
#     seq = ['None']
#     caption = ''
#     prev_caption = ''
#     def recv(self, frame):
#         try:
#             image = frame.to_ndarray(format="bgr24")
#             image = cv2.flip(image, 1)
#             debug_image = copy.deepcopy(image)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             image.flags.writeable = False
#             results = hands.process(image)

#             both_hand_landmarks = []
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                     landmarks = []
#                     for landmark in hand_landmarks.landmark:
#                         landmarks.append((landmark.x, landmark.y))
#                     both_hand_landmarks.append(landmarks)

#                 if len(both_hand_landmarks) == 1:
#                     both_hand_landmarks.append([(0, 0)] * len(both_hand_landmarks[0]))
#                 values = list(np.array(both_hand_landmarks).flatten())
#                 values = scaler.transform([values])
#                 predicted = loaded_model.predict(values)
#                 cv2.rectangle(debug_image, (0, 0), (160, 60), (245, 90, 16), -1)

#                 cv2.putText(debug_image, 'Predicted Gesture', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(debug_image, str(predicted[0]), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 self.threshold_list.append(predicted[0])
#                 if self.threshold_list.count(predicted[0]) >= self.threshold:
#                     if self.seq[-1] != predicted[0]:
#                         self.caption = generate_caption(predicted[0], self.seq)
#                     if self.caption == '':
#                         self.caption = self.prev_caption
#                     else:
#                         self.prev_caption = self.caption
#                     self.threshold_list = []

#             caption_size = cv2.getTextSize(self.caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
#             caption_x = int((debug_image.shape[1] - caption_size[0]) / 2)
#             caption_y = debug_image.shape[0] - 10  # Adjust 10 for padding
#             cv2.putText(debug_image, self.caption, (caption_x, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

#             return av.VideoFrame.from_ndarray(debug_image, format="bgr24")
#         except Exception as e:
#             logger.error(f"Error processing video frame: {e}")
#             return frame


# webrtc_ctx = webrtc_streamer(
#         key="opencv-filter",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,
#         video_processor_factory=VideoProcessor,
#         async_processing=True,
#     )

import logging
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoProcessorBase
import streamlit as st
import av  # type: ignore
import numpy as np
import cv2
import pickle
import mediapipe as mp
import copy
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
confidence = 0.5
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=confidence)

with open('trained_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('trained_models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

try:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
except Exception as e:
    logger.error(f"Error initializing RTCConfiguration: {e}")
    st.error(f"Error initializing RTCConfiguration: {e}")

st.set_page_config(page_title="ISLT", page_icon="🤖")

st.title("Real Time Hand Gesture recognition and Live Hand Sign Translator")
st.text("Developed by Batch - 17")

df = pd.read_csv("landmark_data/Gestures_sentences.csv")
my_dict = df.set_index('gesture_names')['sentence'].to_dict()
final_dict = {}
for key in my_dict:
    t = []
    words = key.split(',')
    for word in words:
        t.append(word)
    s = ' '.join(t)
    final_dict[s] = my_dict[key]

word_limit = 3
def generate_caption(word, seq):
    res = ''
    if len(seq) < word_limit:
        seq.append(word)
        seq.append(word)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]

    elif len(seq) == word_limit:
        seq.pop(0)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]
        seq.append(word)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]
    return res

class VideoProcessor(VideoProcessorBase):
    type: Literal["noop", "cartoon", "edges", "rotate"]

    def __init__(self) -> None:
        self.type = "noop"
    threshold_list = []
    threshold = 20
    seq = ['None']
    caption = ''
    prev_caption = ''
    def recv(self, frame):
        try:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)

            both_hand_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append((landmark.x, landmark.y))
                    both_hand_landmarks.append(landmarks)

                if len(both_hand_landmarks) == 1:
                    both_hand_landmarks.append([(0, 0)] * len(both_hand_landmarks[0]))
                values = list(np.array(both_hand_landmarks).flatten())
                values = scaler.transform([values])
                predicted = loaded_model.predict(values)
                cv2.rectangle(debug_image, (0, 0), (160, 60), (245, 90, 16), -1)

                cv2.putText(debug_image, 'Predicted Gesture', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(debug_image, str(predicted[0]), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                self.threshold_list.append(predicted[0])
                if self.threshold_list.count(predicted[0]) >= self.threshold:
                    if self.seq[-1] != predicted[0]:
                        self.caption = generate_caption(predicted[0], self.seq)
                    if self.caption == '':
                        self.caption = self.prev_caption
                    else:
                        self.prev_caption = self.caption
                    self.threshold_list = []

            caption_size = cv2.getTextSize(self.caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            caption_x = int((debug_image.shape[1] - caption_size[0]) / 2)
            caption_y = debug_image.shape[0] - 10  # Adjust 10 for padding
            cv2.putText(debug_image, self.caption, (caption_x, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(debug_image, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return frame

webrtc_ctx = webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

def main():
    with server_state_lock["webrtc_contexts"]:
        if "webrtc_contexts" not in server_state:
            server_state["webrtc_contexts"] = []

    with server_state_lock["mix_track"]:
        if "mix_track" not in server_state:
            server_state["mix_track"] = create_mix_track(
                kind="video", mixer_callback=mixer_callback, key="mix"
            )

    mix_track = server_state["mix_track"]

    self_ctx = webrtc_streamer(
        key="self",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
        source_video_track=mix_track,
        sendback_audio=False,
    )

    self_process_track = None
    if self_ctx.input_video_track:
        self_process_track = create_process_track(
            input_track=self_ctx.input_video_track,
            processor_factory=OpenCVVideoProcessor,
        )
        mix_track.add_input_track(self_process_track)

        self_process_track.processor.type = st.radio(
            "Select transform type",
            ("noop", "cartoon", "edges", "rotate"),
            key="filter1-type",
        )

    with server_state_lock["webrtc_contexts"]:
        webrtc_contexts: List[WebRtcStreamerContext] = server_state["webrtc_contexts"]
        self_is_playing = self_ctx.state.playing and self_process_track
        if self_is_playing and self_ctx not in webrtc_contexts:
            webrtc_contexts.append(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts
        elif not self_is_playing and self_ctx in webrtc_contexts:
            webrtc_contexts.remove(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts

    if self_ctx.state.playing:
        # Audio streams are transferred in SFU manner
        # TODO: Create MCU to mix audio streams
        for ctx in webrtc_contexts:
            if ctx == self_ctx or not ctx.state.playing:
                continue
            webrtc_streamer(
                key=f"sound-{id(ctx)}",
                mode=WebRtcMode.RECVONLY,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": False, "audio": True},
                source_audio_track=ctx.input_audio_track,
                desired_playing_state=ctx.state.playing,
            )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

    aioice_logger = logging.getLogger("aioice")
    aioice_logger.setLevel(logging.WARNING)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()

