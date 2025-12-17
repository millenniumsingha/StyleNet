"""Streamlit frontend for Fashion MNIST classifier."""

import streamlit as st
import numpy as np
from PIL import Image
import requests
import io
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, MODEL_PATH
from src.predict import FashionClassifier

# Page config
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Load the classifier model (cached)."""
    # Check if model exists, if not train it
    if not MODEL_PATH.exists():
        st.info("⚠️ Model not found. Training for the first time... This may take a minute.")
        with st.spinner("Training model..."):
            from src.train import train_model
            # Train with slightly fewer epochs for faster cloud startup, or full 15
            train_model(model_type='cnn', epochs=10)
            st.success("Training complete!")

    classifier = FashionClassifier()
    classifier.load_model()
    return classifier


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    return "confidence-low"


def main():
    # Header
    st.markdown('<p class="main-header">👗 Fashion MNIST Classifier</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a Convolutional Neural Network (CNN) to classify 
        fashion items into 10 categories.
        
        **Supported Classes:**
        """)
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{i}. {name}")
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.write("""
        - **Architecture**: CNN with 3 conv blocks
        - **Input Size**: 28x28 grayscale
        - **Accuracy**: ~92% on test set
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image of a clothing item",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Upload a grayscale or color image of a clothing item"
        )
        
        # Option to use sample images
        st.markdown("---")
        st.subheader("Or try a sample image:")
        
        use_sample = st.checkbox("Use Fashion MNIST sample")
        sample_index = st.slider("Sample index", 0, 999, 0) if use_sample else 0
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None or use_sample:
            try:
                # Load classifier
                classifier = load_classifier()
                
                if use_sample:
                    # Load sample from Fashion MNIST
                    from tensorflow import keras
                    (_, _), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
                    image_array = test_images[sample_index]
                    true_label = CLASS_NAMES[test_labels[sample_index]]
                    image = Image.fromarray(image_array)
                else:
                    # Load uploaded image
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    true_label = None
                
                # Display image
                st.image(image, caption="Input Image", width=200)
                
                # Get prediction
                with st.spinner("Classifying..."):
                    result = classifier.predict(image_array)
                
                # Display results
                st.markdown("### Prediction")
                confidence = result['confidence']
                conf_class = get_confidence_class(confidence)
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{result['predicted_class']}</h2>
                    <p>Confidence: <span class="{conf_class}">{confidence:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                if true_label:
                    if result['predicted_class'] == true_label:
                        st.success(f"✅ Correct! True label: {true_label}")
                    else:
                        st.error(f"❌ Incorrect. True label: {true_label}")
                
                # Top predictions
                st.markdown("### Top 3 Predictions")
                for pred in result['top_predictions']:
                    progress = pred['confidence']
                    st.write(f"**{pred['class_name']}**")
                    st.progress(progress)
                    st.caption(f"{progress:.1%}")
                
                # All probabilities
                with st.expander("📊 All Class Probabilities"):
                    import pandas as pd
                    probs_df = pd.DataFrame([
                        {"Class": name, "Probability": prob}
                        for name, prob in result['all_probabilities'].items()
                    ]).sort_values("Probability", ascending=False)
                    st.bar_chart(probs_df.set_index("Class"))
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the model is trained. Run: `python -m src.train`")
        else:
            st.info("👈 Upload an image or select a sample to get started!")


if __name__ == "__main__":
    main()
