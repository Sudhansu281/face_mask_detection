import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

@st.cache_resource
def load_model():
    return YOLO("runs/detect/runs/face_mask_detection-4/weights/best.pt")

model = load_model()

st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("Face Mask Detection")
st.write("Detect whether a person is wearing a face mask using an image or webcam.")

option = st.sidebar.selectbox(
    "Choose Detection Mode",
    ["Image Detection", "Webcam Detection"]
)

if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)

        results = model(image_np)

        annotated = results[0].plot()

        st.subheader("Detection Result")

        st.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )
        
elif option == "Webcam Detection":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = None

    if run:

        cap = cv2.VideoCapture(0)

        while run:

            success, frame = cap.read()

            if not success:
                st.error("Could not access webcam.")
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                "Press Stop Checkbox to Exit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            FRAME_WINDOW.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            )

    if cap is not None:
        cap.release()

# Branch check  