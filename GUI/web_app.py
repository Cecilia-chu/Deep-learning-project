import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
os.chdir(os.path.split(__file__)[0])

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io


st.set_page_config(
    page_title="Protein Secondary Structure Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1.5rem;
    }
    .info-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .result-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

import base64
from pathlib import Path


def get_base64_of_image(image_path):
    """Obtain the base64 encoding of the image"""
    return base64.b64encode(Path(image_path).read_bytes()).decode()

def set_background_image(image_path):
    """Set the background picture"""
    base64_image = get_base64_of_image(image_path)
    
    # Create CSS
    page_bg_img = f'''
    <style>
    /* Set the background picture */
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Enhance all text elements */
    .main-header, .sub-header, h1, h2, h3, h4, h5, p, span, li, label, 
    .stMarkdown, .stText, div, .stSelectbox label, .stFileUploader label,
    .stButton, .stDownloadButton, .stMetric, .stDataFrame {{
        text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white;
    }}
    
    /* Special enhancement of ordinary text paragraphs */
    p, span, .stMarkdown p, .stText p, div p, .stMetric {{
        font-weight: 500;
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.5);
        padding: 2px 4px;
        border-radius: 3px;
    }}
    
    /* Make the background of the content block more obvious */
    .section, .home-section, .info-card {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    
    /* Enhance the visibility of the list items */
    li, .bullet-point {{
        margin-bottom: 5px;
        background-color: rgba(255, 255, 255, 0.3);
        padding: 3px 5px;
        border-radius: 3px;
    }}
    </style>
    '''
    
    
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Prediction", "Result Visualization", "SHAP Visualization"])


def load_preprocessing_config():
    
    preprocessing_config = {
        'max_len': 1632,  
        'amino_acids': 'ACDEFGHIKLMNPQRSTVWY',
        'sec_structs': ['H', 'E', 'C']
    }
    return preprocessing_config

# load model
@st.cache_resource
def load_model():
    
    try:
        model = tf.keras.models.load_model('protein_structure_model.h5')
        
        print(f"model input shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Model could not be loaded: {str(e)}")
        
        input_layer = tf.keras.layers.Input(shape=(1632, 21))
        output_layer = tf.keras.layers.Dense(3, activation='softmax')(input_layer)
        dummy_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        print("use model to show")
        return dummy_model

# one-hot
def one_hot_encode_sequences(sequences, max_len=None, amino_acids='ACDEFGHIKLMNPQRSTVWY'):
    aa_to_int = {aa: idx for idx, aa in enumerate(amino_acids)}
    aa_to_int['X'] = len(amino_acids) 
    
    integer_encoded = []
    for seq in sequences:
        encoded = [aa_to_int.get(aa, len(amino_acids)) for aa in seq]
        integer_encoded.append(encoded)
    
    if not max_len:
        max_len = max(len(seq) for seq in integer_encoded)
    
    padded_encoded = tf.keras.preprocessing.sequence.pad_sequences(
        integer_encoded, maxlen=max_len, padding='post', truncating='post', value=len(amino_acids)
    )
    
    one_hot_encoded = to_categorical(padded_encoded, num_classes=len(amino_acids)+1)
    
    return one_hot_encoded, max_len


def predict_secondary_structure(seq, model, config):
    try:
        
        X, _ = one_hot_encode_sequences([seq], max_len=config['max_len'], amino_acids=config['amino_acids'])
        
        
        y_pred = model.predict(X)
        
        
        structure_labels = []
        for pos in range(min(len(seq), config['max_len'])):
            class_idx = np.argmax(y_pred[0, pos])
            structure_labels.append(config['sec_structs'][class_idx])
        
        return ''.join(structure_labels)
    except Exception as e:
        st.error(f"error prediction: {str(e)}")
        return "prediction error"


def load_data_from_csv(file):
    try:
        df = pd.read_csv(file)
        
        if 'seq' not in df.columns:
            st.error("CSV file must contain 'seq' column with protein sequences.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Home
def show_home():
    st.markdown("<h1 class='main-header'>Protein Secondary Structure Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Project Introduction</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>The secondary structure of protein mainly includes α-helices, β-sheets, and random coils. The formation of these structures is determined by the hydrogen bond between the atoms of the protein main chain. Accurately predicting the secondary structure of proteins can help researchers make key breakthroughs in drug design, disease mechanism research, bioengineering and many other fields.</p>", unsafe_allow_html=True)
    
    st.markdown("<p class='info-text'>By using <span class='highlight'>deep learning method</span>, this project can receive the amino acid sequence of the protein input by the user, and after complex calculation and analysis, output the secondary structure prediction result of the protein quickly and accurately. Users do not need to have a deep programming foundation or complex bioinformatics knowledge, just through the simple and intuitive interface operation, it is easy to complete the prediction task.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>How to Use</h2>", unsafe_allow_html=True)
    st.markdown("""
    <ol class='info-text'>
        <li>Users can choose to manually enter the protein sequence or upload the file</li>
        <li>Click the prediction button</li>
        <li>View the results in the results box</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Model Architecture</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='info-text'>Our model employs a sophisticated architecture including:</p>
        <ul class='info-text'>
            <li>Convolutional layers for capturing local patterns</li>
            <li>Bidirectional LSTM for sequential context</li>
            <li>Attention mechanism for focusing on important regions</li>
            <li>Dense layers for final classification</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Secondary Structure Classes</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='info-text'>The model predicts three main secondary structure classes:</p>
        <ul class='info-text'>
            <li><span class='highlight'>H</span>: α-helices</li>
            <li><span class='highlight'>E</span>: β-sheets</li>
            <li><span class='highlight'>C</span>: Coils and other structures</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# prediction
def show_prediction():
    st.markdown("<h1 class='main-header'>Model Prediction</h1>", unsafe_allow_html=True)
    
    # load
    config = load_preprocessing_config()
    model = load_model()
    
    if model is None:
        st.warning("Model is not available. Please check the model file path.")
        return
    
    
    tab1, tab2 = st.tabs(["Sequence Input", "File Upload"])
    
    with tab1:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Enter Protein Sequence</h2>", unsafe_allow_html=True)
        
        
        sequence = st.text_area(
            "Enter amino acid sequence", 
            height=150,
            placeholder="Enter protein sequence using single-letter amino acid codes (e.g. MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT...)",
            help="Enter a protein sequence using single-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
        )
        
        # button
        if st.button("Predict Secondary Structure", key="predict_btn"):
            if sequence:
                
                valid_amino_acids = set(config['amino_acids'])
                invalid_chars = [c for c in sequence.upper() if c not in valid_amino_acids and c != 'X']
                
                if invalid_chars:
                    st.error(f"Invalid characters in sequence: {''.join(set(invalid_chars))}. Please use valid amino acid codes.")
                else:
                    
                    with st.spinner("Predicting secondary structure..."):
                        try:
                            
                            prediction = predict_secondary_structure(sequence.upper(), model, config)
                            
                            if prediction == "prediction error":
                                st.error("error,Please check the input sequence or model configuration")
                                return
                                
                            
                            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                            st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
                            
                            
                            result_df = pd.DataFrame({
                                'Amino Acid': list(sequence.upper()),
                                'Predicted Structure': list(prediction)
                            })
                            
                            # result
                            h_count = prediction.count('H')
                            e_count = prediction.count('E')
                            c_count = prediction.count('C')
                            total = len(prediction)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("α-helices (H)", f"{h_count}", f"{h_count/total:.1%}")
                            col2.metric("β-sheets (E)", f"{e_count}", f"{e_count/total:.1%}")
                            col3.metric("Coils (C)", f"{c_count}", f"{c_count/total:.1%}")
                            
                            
                            st.markdown("<h4>Sequence and Structure:</h4>", unsafe_allow_html=True)
                            st.dataframe(result_df)
                            

                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"error: {str(e)}")
                            st.info("Please ensure that the model has been loaded correctly and that the input sequence format is correct")
            else:
                st.warning("Please enter a protein sequence.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Upload CSV File</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='info-text'>Upload a CSV file containing protein sequences. The file should include a column named 'seq' containing the amino acid sequences.</p>
        """, unsafe_allow_html=True)
        
        # upload file
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # load data
            df = load_data_from_csv(uploaded_file)
            
            if df is not None:
                st.success(f"File uploaded successfully. Found {len(df)} sequences.")
                
                
                st.subheader("Data Preview")
                st.dataframe(df.head(5))
                
                if st.button("Predict Structures for All Sequences", key="predict_csv_btn"):
                    
                    with st.spinner("Processing sequences... This may take a while for large files."):
                        try:
                            
                            df['predicted_structure'] = "Processing..."
                            results = []
                            for i, row in df.iterrows():
                                try:
                                    seq = row['seq'].upper()
                                    pred = predict_secondary_structure(seq, model, config)
                                    results.append(pred)
                                except Exception as e:
                                    st.warning(f"error for line {i+1} sequence: {str(e)}")
                                    results.append("error")
                            
                            df['predicted_structure'] = results
                            
                            
                            successful_preds = df[df['predicted_structure'] != "error"]
                            
                            if len(successful_preds) == 0:
                                st.error("All sequence predictions failed. Please check the sequence format or model configuration")
                                return
                            
                            # result
                            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                            st.markdown("<h3>Prediction Results:</h3>", unsafe_allow_html=True)
                            
                        
                            st.dataframe(df.head(10))
                            
                            
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="protein_structure_predictions.csv",
                                mime="text/csv",
                            )
                            
                            
                            st.markdown("<h4>Summary Statistics:</h4>", unsafe_allow_html=True)
                            
                            
                            structure_stats = []
                            for i, row in successful_preds.iterrows():
                                struct = row['predicted_structure']
                                h_percent = struct.count('H') / len(struct) if len(struct) > 0 else 0
                                e_percent = struct.count('E') / len(struct) if len(struct) > 0 else 0
                                c_percent = struct.count('C') / len(struct) if len(struct) > 0 else 0
                                structure_stats.append([h_percent, e_percent, c_percent])

                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"error occurred when processing the sequence: {str(e)}")
                            st.info("Please ensure that the model has been loaded correctly and that the input data format is correct")
        st.markdown("</div>", unsafe_allow_html=True)


def show_results():
    st.markdown("<h1 class='main-header'>Result Visualization</h1>", unsafe_allow_html=True)
    

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
    
 
    image_paths = {
        "Training Loss": "result image/Loss Curve.png",
        "Training Accuracy": "result image/Accuracy Curve.png",
        "Confusion Matrix": "result image/confusion_matrix.png",
        "ROC Curves": "result image/roc_curves.png"
    }
    
    selected_chart = st.selectbox("Select Chart to Display", list(image_paths.keys()))
    
    image_path = image_paths[selected_chart]
    try:
        image = Image.open(image_path)
        st.image(image, caption=selected_chart, use_container_width=True)
        
    
        if selected_chart == "Training Loss":
            st.markdown("""
            <p class='info-text'>The loss curve shows how the model's error decreased during training. 
            A decreasing trend indicates that the model is learning effectively.</p>
            """, unsafe_allow_html=True)
        elif selected_chart == "Training Accuracy":
            st.markdown("""
            <p class='info-text'>The accuracy curve shows how the model's prediction accuracy improved during training.
            The validation accuracy provides insight into how well the model generalizes to unseen data.</p>
            """, unsafe_allow_html=True)
        elif selected_chart == "Confusion Matrix":
            st.markdown("""
            <p class='info-text'>The confusion matrix illustrates the model's performance across different classes:
            <ul>
                <li>Class 0: α-helices (H)</li>
                <li>Class 1: β-sheets (E)</li>
                <li>Class 2: Coils (C)</li>
            </ul>
            The diagonal elements represent correct predictions, while off-diagonal elements represent misclassifications.</p>
            """, unsafe_allow_html=True)
        elif selected_chart == "ROC Curves":
            st.markdown("""
            <p class='info-text'>The ROC (Receiver Operating Characteristic) curves plot the true positive rate against 
            the false positive rate at various threshold settings. The area under the curve (AUC) quantifies the model's 
            ability to discriminate between classes.</p>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Image file not found: {image_path}")
        st.markdown(f"""
        <p class='info-text'>The selected image file could not be loaded. Please ensure that the 'result image' 
        directory exists and contains the necessary files.</p>
        """, unsafe_allow_html=True)


# SHAP
def show_shap():
    st.markdown("<h1 class='main-header'>SHAP Visualization</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Model Interpretability with SHAP</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>SHAP (SHapley Additive exPlanations) values help us understand how each feature contributes
    to the model's predictions. In the context of protein secondary structure prediction, SHAP values indicate
    which amino acids are most important for predicting specific structure types.</p>
    """, unsafe_allow_html=True)
    
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        
        structure_type = st.radio("Select Structure Type", ["H (α-helices)", "E (β-sheets)", "C (Coils)"])
        structure_code = structure_type[0]
        
        
        plot_type = st.radio("Select Plot Type", ["Summary Dot", "Summary Bar", "Bar Sample", "Waterfall", "Force Plot"])
    
    with col2:
    
        if plot_type == "Summary Dot":
            image_path = f"shap image/{structure_code}_summary_dot.jpg"
        elif plot_type == "Summary Bar":
            image_path = f"shap image/{structure_code}_summary_bar.jpg"
        elif plot_type == "Bar Sample":
            image_path = f"shap image/{structure_code}_bar_sample0.jpg"
        elif plot_type == "Waterfall":
            image_path = f"shap image/{structure_code}_waterfall_sample_0.jpg"
        else:  # Force Plot
            image_path = f"shap image/{structure_code}_force_sample_0.jpg"
        
        try:
            image = Image.open(image_path)
            st.image(image, caption=f"{plot_type} for {structure_type}", use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Image file not found: {image_path}")
            st.markdown(f"""
            <p class='info-text'>The selected SHAP visualization could not be loaded. Please ensure that the 'shap image' 
            directory exists and contains the necessary files.</p>
            """, unsafe_allow_html=True)
    
    
    st.markdown("<h3>SHAP Visualization Explanation</h3>", unsafe_allow_html=True)
    
    if plot_type == "Summary Dot":
        st.markdown("""
        <p class='info-text'>The summary dot plot shows feature importance across all samples. Each point represents a SHAP value
        for a specific feature and sample. Features are ranked by their impact on the model's output, with red indicating higher
        feature values and blue indicating lower values.</p>
        """, unsafe_allow_html=True)
    elif plot_type == "Summary Bar":
        st.markdown("""
        <p class='info-text'>The summary bar plot shows the average absolute SHAP value for each feature, indicating overall
        feature importance. Higher bars represent more influential amino acids for predicting the selected structure type.</p>
        """, unsafe_allow_html=True)
    elif plot_type == "Bar Sample":
        st.markdown("""
        <p class='info-text'>The bar sample plot shows SHAP values for a single example, with positive values pushing the prediction
        higher and negative values pushing it lower. This helps understand how each amino acid contributes to a specific prediction.</p>
        """, unsafe_allow_html=True)
    elif plot_type == "Waterfall":
        st.markdown("""
        <p class='info-text'>The waterfall plot explains how we get from the base value (average model output) to the final prediction
        for a single example, showing the cumulative effect of each feature.</p>
        """, unsafe_allow_html=True)
    else:  # Force Plot
        st.markdown("""
        <p class='info-text'>The force plot shows how features push the prediction from the base value (average model output) toward
        the final prediction. Red features push the prediction higher, while blue features push it lower.</p>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
   
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Analysis of Amino Acid Contributions</h2>", unsafe_allow_html=True)
    
    if structure_code == "H":
        st.markdown("""
        <p class='info-text'>For <span class='highlight'>α-helices (H)</span>, the SHAP analysis reveals:</p>
        <ul class='info-text'>
            <li>Alanine (A), Leucine (L), and Glutamic acid (E) show strong positive contributions, confirming their helix-promoting nature.</li>
            <li>Proline (P) and Glycine (G) typically show negative contributions, consistent with their helix-breaking properties.</li>
            <li>The position context of amino acids matters, as seen in the force plots where identical amino acids may have different impacts depending on their neighbors.</li>
        </ul>
        """, unsafe_allow_html=True)
    elif structure_code == "E":
        st.markdown("""
        <p class='info-text'>For <span class='highlight'>β-sheets (E)</span>, the SHAP analysis reveals:</p>
        <ul class='info-text'>
            <li>Valine (V), Isoleucine (I), and Phenylalanine (F) typically show positive contributions, confirming their sheet-promoting properties.</li>
            <li>Proline (P) shows strong negative contributions due to its rigid structure that disrupts β-sheets.</li>
            <li>Alternating patterns of hydrophobic and hydrophilic residues often increase the likelihood of β-sheet formation.</li>
        </ul>
        """, unsafe_allow_html=True)
    else:  # C
        st.markdown("""
        <p class='info-text'>For <span class='highlight'>Coils (C)</span>, the SHAP analysis reveals:</p>
        <ul class='info-text'>
            <li>Glycine (G) and Proline (P) show positive contributions due to their flexibility and helix-breaking properties.</li>
            <li>Charged residues like Lysine (K) and Arginine (R) often appear important at certain positions in coil regions.</li>
            <li>The presence of multiple consecutive structurally flexible residues increases the likelihood of coil predictions.</li>
        </ul>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# main
def main():
    set_background_image('background.png')
    
    if page == "Home":
        show_home()
    elif page == "Model Prediction":
        show_prediction()
    elif page == "Result Visualization":
        show_results()
    elif page == "SHAP Visualization":
        show_shap()

if __name__ == "__main__":
    main()