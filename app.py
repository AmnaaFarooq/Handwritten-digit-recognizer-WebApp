import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import numpy as np
from PIL import Image

# Load the trained model
# Use the full path so there is no confusion
# model = joblib.load(r'C:\Users\Admin\Desktop\ML project # 01\mnist_model.pkl')
model = joblib.load('mnist_model.pkl')

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the box below!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Get image and convert to grayscale
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L').resize((28, 28))

    # Process the array
    img_array = np.array(img)

    # CRITICAL: If the prediction is still wrong, your colors might be flipped.
    # MNIST expects white lines (255) on a black background (0).
    # If your canvas provides the opposite, uncomment the line below:
    # img_array = 255 - img_array

    # Normalize and Reshape
    img_array = img_array.reshape(1, 784) / 255.0

    if st.button('Predict Number'):
        prediction = model.predict(img_array)
        # Show probabilities so you can see what else the model is thinking
        probs = model.predict_proba(img_array)

        st.header(f"Result: {prediction[0]}")
        st.write(f"Confidence: {np.max(probs) * 100:.2f}%")