import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from src.predictor import TumorPredictor

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classifier Pro",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    """Loads and caches the prediction model."""
    try:
        return TumorPredictor(model_path="brain_tumor_model.keras")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def main():
    st.title("🧠 Brain Tumor Classification System")
    st.markdown("---")

    predictor = get_predictor()
    if not predictor:
        return

    # Two-column layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📁 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose a file (JPG, PNG, JPEG)", 
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Analysis Results")
        if uploaded_file:
            with st.spinner("Analyzing image..."):
                try:
                    results = predictor.predict(image)
                    
                    # Display Classification
                    st.success(f"**Primary Diagnosis:** {results['class']}")
                    
                    # Real-time feedback: Processed image
                    with st.expander("👁️ View Processed Image (CLAHE)"):
                        st.image(results['processed_img'], caption="What the model 'sees' (Normalized + CLAHE)", use_container_width=True)

                    # Probability Breakdown
                    st.markdown("### 📊 Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Condition': list(results['probabilities'].keys()),
                        'Probability': list(results['probabilities'].values())
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Probability', 
                        y='Condition', 
                        orientation='h',
                        text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'),
                        color='Probability',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
        else:
            st.info("Upload an image to start the analysis.")

    # Footer/Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")

if __name__ == "__main__":
    main()
