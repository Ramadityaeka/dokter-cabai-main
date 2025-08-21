import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Dokter Cabai - Sistem Diagnosa Penyakit Tanaman Cabai",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the disease information
with open('disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

# Load the model
@st.cache_resource
def load_classification_model():
    try:
        model = load_model('final_dokter_cabai.h5')  
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    # Convert PIL Image to temporary file
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    # Load using keras method to ensure consistency with training
    img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Remove temporary file
    import os
    os.remove(temp_path)
    return preprocess_input(img_array)


# Predict function
def predict_disease(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Create confidence chart
def create_confidence_chart(prediction, class_names):
    confidences = prediction[0] * 100
    
    # Create a horizontal bar chart
    fig = go.Figure(go.Bar(
        x=confidences,
        y=class_names,
        orientation='h',
        marker=dict(
            color=['#2ecc71' if i == np.argmax(confidences) else '#3498db' for i in range(len(confidences))],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{conf:.1f}%' for conf in confidences],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))
    
    fig.update_layout(
        title="Tingkat Keyakinan Prediksi",
        xaxis_title="Persentase Keyakinan (%)",
        yaxis_title="Kondisi Tanaman",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        title_font=dict(size=16, color='#2c3e50')
    )
    
    return fig

# Main app
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-title {
        font-size: 2.5rem !important;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-subtitle {
        font-size: 1.2rem !important;
        opacity: 0.9;
        margin-bottom: 0;
    }
    .feature-card {
        linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .result-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    .result-title {
        font-size: 1.8rem !important;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.2rem !important;
        opacity: 0.9;
    }
    .info-section {
        background: black;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .info-header {
        font-size: 1.4rem !important;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .characteristic-item {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(116, 185, 255, 0.3);
    }
    .prevention-item {
        background: linear-gradient(135deg, #00cec9 0%, #00b894 100%);
        color: white;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 206, 201, 0.3);
    }
    .solution-item {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(253, 121, 168, 0.3);
    }
    .upload-area {
        border: 3px dashed #3498db;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(155, 89, 182, 0.1) 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #2980b9;
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.15) 0%, rgba(155, 89, 182, 0.15) 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .source-link {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        border-left: 3px solid #3498db;
        font-size: 0.9rem;
    }
    .source-link a {
        color: #2980b9;
        text-decoration: none;
    }
    .source-link a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🌶️ Dokter Cabai</div>
        <div class="main-subtitle">Sistem Diagnosa Penyakit Tanaman Cabai Berbasis AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h3>📊 Statistik Model</h3>
            <p>Model: MobileNetV2</p>
            <p>Akurasi: 83%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Kondisi yang Dapat Dideteksi:")
        conditions = [
            "🟢 Daun Sehat",
            "🟡 Keriting Daun (Leaf Curl)",
            "🔴 Bercak Daun (Leaf Spot)",
            "⚪ Serangan Kutu Kebul",
            "🟨 Daun Menguning"
        ]
        for condition in conditions:
            st.markdown(f"- {condition}")
            

    # Introduction
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🌱 Praktis Digunakan</h4>
            <p>Cukup unggah foto daun cabai, sistem langsung menganalisis</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Cepat & Akurat</h4>
            <p>Prediksi penyakit dilakukan dengan model AI MobileNetV2 berakurasi tinggi</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📖 Informasi Edukatif</h4>
            <p>Hasil diagnosis dilengkapi deskripsi, pencegahan, dan solusi penanganan</p>
        </div>
        """, unsafe_allow_html=True)

    # Load model
    model = load_classification_model()
    if model is None:
        st.error("⚠️ Gagal memuat model. Silakan coba lagi nanti.")
        return

    # Class names
    class_names = ['Sehat', 'Keriting Daun', 'Bercak Daun', 'Kutu Kebul', 'Menguning']
    class_keys = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

    # Image upload
    st.markdown("""
    <div class="upload-area">
        <h3>📷 Unggah Foto Daun Cabai</h3>
        <p>Pilih gambar daun cabai yang ingin didiagnosa (JPG, JPEG, PNG)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", 
                                    type=["jpg", "jpeg", "png"],
                                    help="Pastikan foto memiliki pencahayaan yang baik dan fokus pada daun")

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🖼️ Gambar yang Diunggah")
                st.image(image, caption='Foto daun cabai', use_column_width=True)

            # Make prediction
            with st.spinner('🔄 Menganalisis gambar...'):
                prediction = predict_disease(model, image)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class_idx]
                predicted_class_key = class_keys[predicted_class_idx]
                predicted_class_name = class_names[predicted_class_idx]

            with col2:
                # Display results
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">🎯 Hasil Diagnosa</div>
                    <div style="font-size: 1.4rem; margin: 0.5rem 0;">
                        <strong>{disease_info[predicted_class_key]['name']}</strong>
                    </div>
                    <div class="confidence-text">
                        Tingkat Keyakinan: {confidence*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                fig = create_confidence_chart(prediction, class_names)
                st.plotly_chart(fig, use_container_width=True)

            # Disease information tabs
            st.markdown("---")
            st.markdown("## 📋 Informasi Lengkap Diagnosa")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📖 Deskripsi", "🔍 Karakteristik", "🛡️ Pencegahan", "💊 Solusi Pengobatan"])
            
            with tab1:
                st.markdown(f"""
                <div class="info-section">
                    <div class="info-header">{disease_info[predicted_class_key]['name']}</div>
                    <p style="font-size: 1.1rem; line-height: 1.6; text-align: justify;">
                        {disease_info[predicted_class_key]['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown("### 🔍 Karakteristik yang Diamati:")
                for char in disease_info[predicted_class_key]['characteristics']:
                    st.markdown(f'<div class="characteristic-item">📍 {char}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown("### 🛡️ Langkah Pencegahan:")
                for prev in disease_info[predicted_class_key]['prevention']:
                    st.markdown(f'<div class="prevention-item">✅ {prev}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown("### 💊 Solusi Pengobatan:")
                for solution in disease_info[predicted_class_key]['solutions']:
                    st.markdown(f'<div class="solution-item">🔬 {solution}</div>', unsafe_allow_html=True)
                
                st.markdown("### 📚 Sumber Referensi:")
                for source in disease_info[predicted_class_key]['sources']:
                    st.markdown(f'<div class="source-link">🔗 <a href="{source["url"]}" target="_blank">{source["title"]}</a></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error memproses gambar: {str(e)}")

    # Additional information
    st.markdown("---")
    
    with st.expander("ℹ️ Tentang Sistem Dokter Cabai"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔬 Teknologi yang Digunakan
            - **Deep Learning**: MobileNetV2 Architecture  
            - **Dataset**: 500+ gambar daun cabai berlabel
            - **Akurasi Model**: 83% pada data testing
            - **Preprocessing**: Image augmentation & normalization
            
            ### 📊 Kondisi yang Dapat Dideteksi
            - **Daun Sehat**: Kondisi normal tanaman cabai
            - **Keriting Daun**: Disebabkan virus Gemini (PYLCV)
            - **Bercak Daun**: Infeksi jamur Cercospora capsici
            - **Kutu Kebul**: Serangan Bemisia tabaci
            - **Daun Menguning**: Defisiensi nutrisi atau masalah drainase
            """)
            
        with col2:
            st.markdown("""
            ### 📸 Tips Penggunaan Optimal
            - Gunakan pencahayaan natural yang cukup
            - Fokuskan kamera pada area daun yang bermasalah
            - Hindari bayangan atau pantulan cahaya berlebihan
            - Pastikan daun mengisi sebagian besar frame foto
            - Gunakan background yang kontras dengan daun
            
            ### ⚠️ Disclaimer
            Hasil diagnosa merupakan prediksi berdasarkan AI dan sebaiknya 
            dikonfirmasi dengan ahli pertanian untuk penanganan yang optimal.
            Sistem ini dirancang sebagai alat bantu, bukan pengganti konsultasi profesional.
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
        <p>🌶️ <strong>Dokter Cabai</strong> - Sistem Diagnosa Penyakit Tanaman Cabai</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()