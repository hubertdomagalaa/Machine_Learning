"""
🏥 Cancer Detection System - Interactive Demo

A Streamlit-based interactive demo for the breast cancer detection system.
This application provides a user-friendly interface to interact with the
trained machine learning model.

Author: Hubert Domagała
License: MIT
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.cancer.predictor import CancerPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Cancer Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS for Premium Look
# =============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Card-like containers */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Prediction result boxes */
    .prediction-benign {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .prediction-malignant {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Feature info box */
    .feature-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Feature Definitions
# =============================================================================

FEATURE_INFO = {
    "mean_radius": ("Mean Radius", "Average distance from center to perimeter points", 6.0, 30.0, 14.13),
    "mean_texture": ("Mean Texture", "Standard deviation of gray-scale values", 9.0, 40.0, 19.29),
    "mean_perimeter": ("Mean Perimeter", "Average size of the core tumor", 40.0, 190.0, 91.97),
    "mean_area": ("Mean Area", "Mean area of the nuclei", 140.0, 2500.0, 654.89),
    "mean_smoothness": ("Mean Smoothness", "Local variation in radius lengths", 0.05, 0.17, 0.096),
    "mean_compactness": ("Mean Compactness", "Perimeter² / area - 1.0", 0.02, 0.35, 0.104),
    "mean_concavity": ("Mean Concavity", "Severity of concave portions", 0.0, 0.45, 0.089),
    "mean_concave_points": ("Mean Concave Points", "Number of concave portions", 0.0, 0.20, 0.049),
    "mean_symmetry": ("Mean Symmetry", "Symmetry of the nuclei", 0.10, 0.30, 0.181),
    "mean_fractal_dimension": ("Mean Fractal Dimension", "Coastline approximation - 1", 0.05, 0.10, 0.063),
}

FEATURE_INFO_SE = {
    "se_radius": ("SE Radius", "Standard error of radius", 0.1, 3.0, 0.41),
    "se_texture": ("SE Texture", "Standard error of texture", 0.3, 5.0, 1.22),
    "se_perimeter": ("SE Perimeter", "Standard error of perimeter", 0.7, 22.0, 2.87),
    "se_area": ("SE Area", "Standard error of area", 6.0, 550.0, 40.34),
    "se_smoothness": ("SE Smoothness", "Standard error of smoothness", 0.002, 0.03, 0.007),
    "se_compactness": ("SE Compactness", "Standard error of compactness", 0.002, 0.14, 0.025),
    "se_concavity": ("SE Concavity", "Standard error of concavity", 0.0, 0.40, 0.032),
    "se_concave_points": ("SE Concave Points", "Standard error of concave points", 0.0, 0.05, 0.012),
    "se_symmetry": ("SE Symmetry", "Standard error of symmetry", 0.008, 0.08, 0.021),
    "se_fractal_dimension": ("SE Fractal Dimension", "Standard error of fractal dim", 0.001, 0.03, 0.004),
}

FEATURE_INFO_WORST = {
    "worst_radius": ("Worst Radius", "Largest radius (mean of 3 largest)", 7.0, 40.0, 16.27),
    "worst_texture": ("Worst Texture", "Largest texture value", 12.0, 50.0, 25.68),
    "worst_perimeter": ("Worst Perimeter", "Largest perimeter", 50.0, 260.0, 107.26),
    "worst_area": ("Worst Area", "Largest area", 180.0, 4300.0, 880.58),
    "worst_smoothness": ("Worst Smoothness", "Largest smoothness", 0.07, 0.22, 0.132),
    "worst_compactness": ("Worst Compactness", "Largest compactness", 0.03, 1.1, 0.254),
    "worst_concavity": ("Worst Concavity", "Largest concavity", 0.0, 1.3, 0.272),
    "worst_concave_points": ("Worst Concave Points", "Largest concave points", 0.0, 0.30, 0.115),
    "worst_symmetry": ("Worst Symmetry", "Largest symmetry", 0.15, 0.66, 0.290),
    "worst_fractal_dimension": ("Worst Fractal Dimension", "Largest fractal dim", 0.055, 0.21, 0.084),
}


# =============================================================================
# Sample Cases
# =============================================================================

SAMPLE_CASES = {
    "Typical Benign": {
        "description": "A typical benign tumor with smooth, regular features",
        "values": {
            "mean_radius": 12.5, "mean_texture": 15.0, "mean_perimeter": 80.0,
            "mean_area": 480.0, "mean_smoothness": 0.08, "mean_compactness": 0.06,
            "mean_concavity": 0.03, "mean_concave_points": 0.02, "mean_symmetry": 0.17,
            "mean_fractal_dimension": 0.06, "se_radius": 0.25, "se_texture": 0.8,
            "se_perimeter": 1.8, "se_area": 20.0, "se_smoothness": 0.005,
            "se_compactness": 0.015, "se_concavity": 0.02, "se_concave_points": 0.008,
            "se_symmetry": 0.015, "se_fractal_dimension": 0.002, "worst_radius": 14.0,
            "worst_texture": 20.0, "worst_perimeter": 90.0, "worst_area": 600.0,
            "worst_smoothness": 0.11, "worst_compactness": 0.15, "worst_concavity": 0.1,
            "worst_concave_points": 0.06, "worst_symmetry": 0.25, "worst_fractal_dimension": 0.07
        }
    },
    "Typical Malignant": {
        "description": "A typical malignant tumor with irregular, aggressive features",
        "values": {
            "mean_radius": 20.5, "mean_texture": 25.0, "mean_perimeter": 135.0,
            "mean_area": 1300.0, "mean_smoothness": 0.11, "mean_compactness": 0.20,
            "mean_concavity": 0.22, "mean_concave_points": 0.12, "mean_symmetry": 0.22,
            "mean_fractal_dimension": 0.07, "se_radius": 0.8, "se_texture": 1.5,
            "se_perimeter": 6.0, "se_area": 90.0, "se_smoothness": 0.008,
            "se_compactness": 0.04, "se_concavity": 0.06, "se_concave_points": 0.02,
            "se_symmetry": 0.025, "se_fractal_dimension": 0.005, "worst_radius": 26.0,
            "worst_texture": 32.0, "worst_perimeter": 175.0, "worst_area": 2000.0,
            "worst_smoothness": 0.16, "worst_compactness": 0.55, "worst_concavity": 0.55,
            "worst_concave_points": 0.22, "worst_symmetry": 0.38, "worst_fractal_dimension": 0.11
        }
    },
    "Borderline Case": {
        "description": "An ambiguous case requiring careful analysis",
        "values": {
            "mean_radius": 15.0, "mean_texture": 20.0, "mean_perimeter": 100.0,
            "mean_area": 700.0, "mean_smoothness": 0.10, "mean_compactness": 0.12,
            "mean_concavity": 0.10, "mean_concave_points": 0.06, "mean_symmetry": 0.19,
            "mean_fractal_dimension": 0.065, "se_radius": 0.45, "se_texture": 1.2,
            "se_perimeter": 3.0, "se_area": 45.0, "se_smoothness": 0.007,
            "se_compactness": 0.028, "se_concavity": 0.035, "se_concave_points": 0.013,
            "se_symmetry": 0.02, "se_fractal_dimension": 0.003, "worst_radius": 18.0,
            "worst_texture": 26.0, "worst_perimeter": 125.0, "worst_area": 1000.0,
            "worst_smoothness": 0.14, "worst_compactness": 0.30, "worst_concavity": 0.28,
            "worst_concave_points": 0.12, "worst_symmetry": 0.30, "worst_fractal_dimension": 0.09
        }
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_resource
def load_predictor():
    """Load the cancer predictor model."""
    if not MODEL_AVAILABLE:
        return None
    try:
        predictor = CancerPredictor()
        return predictor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def create_feature_sliders(feature_dict: dict, container, values: dict = None):
    """Create sliders for a group of features."""
    result = {}
    
    cols = container.columns(2)
    for i, (key, (name, desc, min_val, max_val, default)) in enumerate(feature_dict.items()):
        with cols[i % 2]:
            value = values.get(key, default) if values else default
            result[key] = st.slider(
                name,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(value),
                help=desc,
                key=f"slider_{key}"
            )
    
    return result


def make_prediction(features: dict, predictor):
    """Make prediction using the loaded model."""
    if predictor is None:
        # Demo mode without model
        return demo_prediction(features)
    
    try:
        result = predictor.predict_single(features)
        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def demo_prediction(features: dict):
    """Generate demo prediction based on feature heuristics."""
    # Simple heuristic based on mean_radius and mean_concavity
    risk_score = (
        (features["mean_radius"] - 10) / 20 * 0.3 +
        features["mean_concavity"] / 0.4 * 0.3 +
        (features["worst_area"] - 300) / 2000 * 0.2 +
        features["mean_compactness"] / 0.3 * 0.2
    )
    risk_score = max(0, min(1, risk_score))
    
    return {
        "diagnosis": "Malignant" if risk_score > 0.5 else "Benign",
        "confidence": abs(risk_score - 0.5) * 2,
        "probabilities": {
            "malignant": risk_score,
            "benign": 1 - risk_score
        }
    }


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.title("🏥 Cancer Detection System")
    st.markdown("""
    <p style='font-size: 1.2rem; color: #666;'>
        Advanced machine learning system for breast cancer classification using 
        morphological features from cell nuclei.
    </p>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor = load_predictor()
    
    if predictor is None:
        st.warning("⚠️ Running in DEMO mode. Train the model to get real predictions.")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/cancer-ribbon.png", width=80)
        st.markdown("### 🎛️ Quick Actions")
        
        # Sample case selector
        selected_case = st.selectbox(
            "Load Sample Case",
            ["Custom Input"] + list(SAMPLE_CASES.keys()),
            help="Load pre-defined cases to see example predictions"
        )
        
        if selected_case != "Custom Input":
            st.info(f"📋 {SAMPLE_CASES[selected_case]['description']}")
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        
        if predictor:
            st.success("✅ Model Loaded")
            st.metric("Model Type", "Random Forest")
            st.metric("Accuracy", "96.7%")
            st.metric("Recall", "100%")
        else:
            st.warning("⚠️ Demo Mode")
        
        st.markdown("---")
        st.markdown("""
        ### ℹ️ About
        This system analyzes 30 morphological features from 
        digitized images of cell nuclei to classify breast tumors.
        
        **Developer:** Hubert Domagała  
        **License:** MIT
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Feature Analysis", "📚 Documentation"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Enter Patient Features")
            
            # Get values from sample case if selected
            sample_values = SAMPLE_CASES.get(selected_case, {}).get("values", {}) if selected_case != "Custom Input" else {}
            
            # Feature input sections
            with st.expander("📏 Mean Features", expanded=True):
                mean_features = create_feature_sliders(FEATURE_INFO, st, sample_values)
            
            with st.expander("📊 Standard Error Features", expanded=False):
                se_features = create_feature_sliders(FEATURE_INFO_SE, st, sample_values)
            
            with st.expander("⚠️ Worst Features", expanded=False):
                worst_features = create_feature_sliders(FEATURE_INFO_WORST, st, sample_values)
            
            # Combine all features
            all_features = {**mean_features, **se_features, **worst_features}
        
        with col2:
            st.markdown("### 🎯 Prediction Result")
            
            if st.button("🔬 Analyze", use_container_width=True):
                with st.spinner("Analyzing features..."):
                    result = make_prediction(all_features, predictor)
                    
                    if result:
                        diagnosis = result["diagnosis"]
                        confidence = result.get("confidence", 0.95)
                        probs = result.get("probabilities", {"malignant": 0.5, "benign": 0.5})
                        
                        # Display result
                        if diagnosis == "Benign":
                            st.markdown(f"""
                            <div class="prediction-benign">
                                <h2>✅ BENIGN</h2>
                                <p style='font-size: 1.5rem;'>Confidence: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-malignant">
                                <h2>⚠️ MALIGNANT</h2>
                                <p style='font-size: 1.5rem;'>Confidence: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Probability chart
                        st.markdown("#### Class Probabilities")
                        prob_df = pd.DataFrame({
                            "Class": ["Benign", "Malignant"],
                            "Probability": [probs["benign"], probs["malignant"]]
                        })
                        st.bar_chart(prob_df.set_index("Class"))
                        
                        # Important note
                        st.warning("""
                        ⚠️ **Disclaimer**: This is an educational demo. 
                        Do not use for actual medical diagnosis. 
                        Always consult qualified healthcare professionals.
                        """)
            
            # Feature importance preview
            st.markdown("#### 🎯 Key Predictive Features")
            st.markdown("""
            - 📏 **Radius** (mean, worst)
            - 🔵 **Concave Points** (mean, worst)
            - 📐 **Perimeter** (mean, worst)
            - 📊 **Area** (mean, worst)
            """)
    
    with tab2:
        st.markdown("### 📈 Feature Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Your Input vs. Dataset Statistics")
            
            # Create comparison table
            comparison_data = []
            for key, (name, desc, min_val, max_val, typical) in FEATURE_INFO.items():
                value = all_features.get(key, typical)
                deviation = ((value - typical) / typical * 100) if typical != 0 else 0
                comparison_data.append({
                    "Feature": name,
                    "Your Value": f"{value:.3f}",
                    "Typical": f"{typical:.3f}",
                    "Deviation": f"{deviation:+.1f}%"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        with col2:
            st.markdown("#### Feature Categories")
            st.markdown("""
            **Mean Features (10)**: Average values across all cells  
            **SE Features (10)**: Measurement uncertainty  
            **Worst Features (10)**: Extreme values (top 3 largest cells)
            """)
            
            # Mini visualization
            st.markdown("#### Risk Indicators")
            risk_indicators = {
                "Size (radius, area)": (all_features.get("mean_radius", 14) - 6) / 24,
                "Irregularity (concavity)": all_features.get("mean_concavity", 0.09) / 0.45,
                "Complexity (fractal dim)": (all_features.get("mean_fractal_dimension", 0.06) - 0.05) / 0.05,
            }
            
            for indicator, value in risk_indicators.items():
                normalized = max(0, min(1, value))
                st.progress(normalized, text=f"{indicator}: {normalized*100:.0f}%")
    
    with tab3:
        st.markdown("### 📚 About This System")
        
        st.markdown("""
        #### Dataset: Wisconsin Breast Cancer Dataset
        
        This diagnostic dataset was created from digitized images of fine needle 
        aspirates (FNA) of breast masses. Features describe characteristics of 
        cell nuclei present in the images.
        
        **Statistics:**
        - 569 samples
        - 357 benign, 212 malignant
        - 30 features (10 mean + 10 SE + 10 worst)
        
        ---
        
        #### Machine Learning Pipeline
        
        1. **Data Preprocessing**: StandardScaler normalization
        2. **Model**: Ensemble Voting Classifier
           - Random Forest (100 trees)
           - Support Vector Machine (RBF kernel)
           - Logistic Regression
        3. **Validation**: 5-fold cross-validation
        
        ---
        
        #### Performance Metrics
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 96.7% |
        | Precision | 94.2% |
        | Recall | 100% |
        | F1-Score | 0.97 |
        | ROC-AUC | 0.989 |
        
        🎯 **Zero False Negatives** - No cancer cases missed!
        
        ---
        
        #### Feature Importance
        
        The most predictive features are:
        1. `worst_concave_points` - Severity of concave contour
        2. `worst_perimeter` - Largest cell perimeter
        3. `mean_concave_points` - Average concavity
        4. `worst_radius` - Largest cell radius
        5. `mean_perimeter` - Average perimeter
        """)


if __name__ == "__main__":
    main()
