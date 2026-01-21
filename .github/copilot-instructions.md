# EEG Web Demo

## Project Overview
This is an EEG signal classification web application that processes brain wave data to predict cognitive states (e.g., eye-open/eye-closed conditions). The system uses machine learning to analyze EEG signals through power spectral density (PSD) and coherence features.

## Architecture & Data Flow

### Core Components
- **`app.py`**: Streamlit web interface for uploading EEG files and displaying predictions
- **`predict.py`**: Main prediction pipeline - loads data, preprocesses, extracts features, runs ML model
- **`utils/autopreprocessing.py`**: Comprehensive EEG preprocessing library with artifact detection and filtering
- **`utils/autopreprocess_pipeline.py`**: Batch preprocessing pipeline for multiple subjects
- **`utils/inout.py`**: File I/O utilities for BIDS-formatted EEG data

### Data Processing Pipeline
1. **Input**: CSV files with 26 EEG channels + auxiliary channels (500 Hz sampling)
2. **Preprocessing**: Filtering (0.5-100 Hz bandpass, 50 Hz notch), artifact removal (EMG, EOG, jumps)
3. **Feature Extraction**: 
   - PSD: Power spectral density (1-45 Hz, log-transformed)
   - Coherence: Cross-channel coherence in 5 frequency bands (delta/theta/alpha/beta/gamma)
4. **Classification**: TensorFlow model with multi-input architecture (PSD + coherence + tabular features)
5. **Output**: Subject-level predictions aggregated from epoch-level classifications

## Key Conventions & Patterns

### EEG Data Format
- **Channels**: 26 EEG channels (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, FC1, FC2, CP1, CP2, FC5, FC6, CP5, CP6)
- **Sampling**: 500 Hz
- **Epochs**: 2-second segments for feature extraction
- **File Structure**: BIDS format (`sub-XXXXXXXX/ses-XX/eeg/sub-XXXXXXXX_ses-XX_task-XX_eeg.csv`)

### Model Input Structure
```python
inputs = {
    "psd": X_psd,      # (n_epochs, 26, 89, 1) - PSD features
    "coh": X_coh,      # (n_epochs, 26, 26, 5) - Coherence matrices
    "cont": X_cont,    # (n_epochs, 3) - continuous vars (age, education, sleep)
    "cat": X_cat       # (n_epochs, 2) - categorical vars (gender, well-being)
}
```

### Preprocessing Standards
- Use `dataset` class from `autopreprocessing.py` for all EEG processing
- Always apply: high-pass (0.5 Hz), low-pass (100 Hz), notch (50 Hz)
- Detect and mark artifacts: EMG, EOG, jumps, kurtosis, bridging
- Segment into 2-second epochs with 50% overlap for feature extraction

## Critical Workflows

### Running Predictions
```bash
# Install dependencies
pip install -r requirements.txt

# Place model artifacts in artifacts/ folder:
# - model_eo_final.keras (TensorFlow model)
# - scaler.pkl (feature scaler)
# - label_encoder.pkl (class encoder)
# - cat_maps.json (categorical mappings)

# Run prediction on CSV file
python predict.py  # Edit the main block with your CSV path and subject info
```

### Batch Preprocessing
```python
from utils.autopreprocess_pipeline import autopreprocess_standard

config = {
    'sourcepath': 'path/to/raw/data',
    'preprocpath': 'path/to/output',
    'condition': ['EO', 'EC']  # or 'all'
}

autopreprocess_standard(config, subject='sub-19681349')
```

### Web App Development
```bash
streamlit run app.py
```
- Upload CSV files through web interface
- Input subject metadata (age, gender, education, sleep quality, well-being)
- View epoch-level and subject-level predictions

## Dependencies & Environment
- **Python 3.8+** required
- **TensorFlow** for model inference (not training)
- **Streamlit** for web interface
- **SciPy** for signal processing (Welch, CSD, filtering)
- **Joblib** for loading pickled scalers/models

## File Organization
- **`artifacts/`**: Trained models, scalers, encoders, mappings
- **`data/`**: Input EEG CSV files (BIDS format)
- **`utils/`**: Preprocessing and I/O utilities
- **Root**: Main application files (`app.py`, `predict.py`)

## Common Patterns
- **Feature Engineering**: Always combine PSD + coherence features
- **Aggregation**: Use mean probabilities for subject-level predictions over majority voting
- **Error Handling**: Check for insufficient data length (< 2 seconds) before processing
- **Logging**: Print progress for long-running preprocessing/feature extraction steps

## Integration Points
- **External Models**: TensorFlow SavedModel format in `artifacts/model_eo_final.keras`
- **Data Sources**: CSV files with Brainclinics Diagnostics format (26 EEG + auxiliary channels)
- **Web Interface**: Streamlit for file upload and result visualization
- **Preprocessing**: Modular design allows swapping preprocessing steps while maintaining feature extraction compatibility</content>
<parameter name="filePath">d:\eeg_web_demo\.github\copilot-instructions.md