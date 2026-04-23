import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from src.predictor import TumorPredictor

# LAYOUT CHANGE: Page Configuration - Updated page_title and layout
st.set_page_config(
    page_title="NeuroScan AI: Brain Tumor Classification", # LAYOUT CHANGE
    page_icon="🧠",
    layout="wide"
)

# LAYOUT CHANGE: Custom CSS for modern dark theme and styled cards
st.markdown("""
    <style>
    /* LAYOUT CHANGE: Main background color for a dark theme */
    .main {
        background-color: #1a1a2e; /* Dark blue/purple background */
        color: #e0e0e0; /* Light grey text */
        font-family: 'Inter', sans-serif; /* LAYOUT CHANGE: Professional font */
    }
    /* LAYOUT CHANGE: Streamlit headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
    }
    /* LAYOUT CHANGE: Card container styling */
    .stCard {
        background-color: #2a2a4a; /* Slightly lighter dark blue for cards */
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); /* Subtle shadow */
        padding: 20px;
        margin-bottom: 20px; /* Spacing between cards */
        border: 1px solid #3a3a5a; /* Subtle border */
    }
    /* LAYOUT CHANGE: Prominent Disclaimer Banner styling */
    .disclaimer-banner {
        background-color: #ffcccc; /* Light red background */
        color: #cc0000; /* Dark red text */
        border-left: 8px solid #cc0000; /* Red border on the left */
        padding: 15px;
        margin-bottom: 30px;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .disclaimer-banner .icon {
        font-size: 1.5em;
        margin-right: 10px;
    }
    /* LAYOUT CHANGE: Info card styling */
    .info-card {
        background-color: #2a2a4a;
        border-radius: 10px;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        min-height: 150px; /* Ensure consistent card height */
        border: 1px solid #3a3a5a;
    }
    .info-card h3 {
        color: #8be9fd; /* Accent color for card titles */
        margin-top: 0;
        font-size: 1.2em;
    }
    .info-card p {
        color: #c0c0c0;
        font-size: 0.9em;
    }
    /* LAYOUT CHANGE: Button styling */
    .stButton>button {
        background-color: #6272a4; /* Accent color for buttons */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4a5a7f;
    }
    /* LAYOUT CHANGE: File uploader styling */
    .stFileUploader {
        background-color: #2a2a4a;
        border-radius: 10px;
        padding: 15px;
        border: 1px dashed #6272a4;
    }
    .stAlert {
        border-radius: 10px;
        background-color: #3a3a5a;
        color: #e0e0e0;
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
    # LAYOUT CHANGE: App Name displayed prominently
    st.markdown("<h1 style='text-align: center; color: #8be9fd;'>NeuroScan AI</h1>", unsafe_allow_html=True) # LAYOUT CHANGE: App Name
    st.markdown("<h3 style='text-align: center; color: #e0e0e0;'>Brain Tumor Classification System</h3>", unsafe_allow_html=True) # LAYOUT CHANGE: Subtitle
    st.markdown("---") # LAYOUT CHANGE: Separator

    # LAYOUT CHANGE: Prominent Disclaimer Banner
    st.markdown("""
        <div class="disclaimer-banner">
            <span class="icon">⚠️</span>
            <span><b>Disclaimer:</b> This is a diagnostic support tool only. Always consult a qualified health professional before making any medical decisions.</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---") # LAYOUT CHANGE: Separator

    # LAYOUT CHANGE: Info Card Section
    st.markdown("<h2 style='text-align: center; color: #e0e0e0;'>About NeuroScan AI</h2>", unsafe_allow_html=True) # LAYOUT CHANGE: Section Title
    col1, col2, col3, col4 = st.columns(4) # LAYOUT CHANGE: Grid for info cards

    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🧠 About Tumours</h3>
                <p>Brain tumors are abnormal growths of cells in the brain. Early detection is crucial for effective treatment and improved patient outcomes. This tool aims to assist in preliminary identification.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>💡 Technology</h3>
                <p>NeuroScan AI utilizes advanced deep learning algorithms, specifically Convolutional Neural Networks (CNNs), trained on vast datasets of medical imaging to identify tumor patterns.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>🎯 Accuracy</h3>
                <p>Our model is trained on validated medical imaging datasets, achieving high accuracy in classifying different types of brain tumors. Performance metrics are continuously monitored and improved.</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="info-card">
                <h3>📋 What This App Does</h3>
                <p>Upload an MRI image, and our AI will analyze it to predict the presence and type of brain tumor, providing a probability breakdown for different conditions.</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---") # LAYOUT CHANGE: Separator

    predictor = get_predictor()
    if not predictor:
        return

    # LAYOUT CHANGE: Main content area with original two-column layout wrapped in a styled container
    st.markdown("<div class='stCard'>", unsafe_allow_html=True) # LAYOUT CHANGE: Wrap main content in a card
    st.markdown("<h2 style='color: #e0e0e0;'>Upload for Analysis</h2>", unsafe_allow_html=True) # LAYOUT CHANGE: Section title
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
                    fig.update_layout(showlegend=False, height=300,
                                      plot_bgcolor='rgba(0,0,0,0)', # LAYOUT CHANGE: Transparent background for plot
                                      paper_bgcolor='rgba(0,0,0,0)', # LAYOUT CHANGE: Transparent background for paper
                                      font_color='#e0e0e0' # LAYOUT CHANGE: Light font color for plot
                                     )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
        else:
            st.info("Upload an image to start the analysis.")
    st.markdown("</div>", unsafe_allow_html=True) # LAYOUT CHANGE: Close main content card

    # LAYOUT CHANGE: Footer/Disclaimer - Repositioned and restyled for prominence
    st.markdown("---")
    st.markdown("""
        <div class="disclaimer-banner">
            <span class="icon">⚠️</span>
            <span><b>Disclaimer:</b> This is a diagnostic support tool only. Always consult a qualified health professional before making any medical decisions.</span>
        </div>
    """, unsafe_allow_html=True) # LAYOUT CHANGE: Disclaimer at the bottom too

if __name__ == "__main__":
    main()
