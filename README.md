# Medical Diagnosis System Using Self-Attention and Multi-Head Attention

## Overview

This project demonstrates a medical diagnosis system that uses attention mechanisms to analyze patient symptoms and vital signs to predict potential diseases. The system provides interpretable attention weights that show which symptoms and measurements are most relevant for each diagnosis, making it valuable for medical professionals.

## Features

- **Self-Attention Mechanism**: Single-head attention for analyzing symptom relationships
- **Multi-Head Attention**: Captures multiple diagnostic reasoning paths simultaneously
- **Medical Feature Encoding**: Specialized encoding for symptoms and vital signs
- **Diagnosis System**: Predicts diseases based on patient data with probability scores
- **Attention Visualization**: Heatmaps showing what the model focuses on for each diagnosis
- **Clinical Insights**: Detailed analysis of attention patterns and diagnostic reasoning

## Architecture

The system consists of four main components (all implemented as functions):

1. **Self-Attention Functions**
   - `softmax()` - Computes attention weights
   - `scaled_dot_product_attention()` - Core attention mechanism
   - `self_attention_forward()` - Single-head attention forward pass

2. **Multi-Head Attention Functions**
   - `split_heads()` - Distributes input across multiple attention heads
   - `concat_heads()` - Combines outputs from multiple heads
   - `multi_head_attention_forward()` - Multi-head attention forward pass

3. **Medical Feature Encoder Functions**
   - `initialize_encoders()` - Creates encoding matrices for symptoms and vitals
   - `encode_symptom()` - Encodes individual symptoms
   - `encode_vital()` - Encodes vital signs with normalization
   - `encode_patient_data()` - Processes complete patient information

4. **Diagnosis System Functions**
   - `initialize_diagnosis_system()` - Initializes system weights and parameters
   - `diagnose()` - Performs diagnosis and returns probabilities with attention weights

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the medical diagnosis system:

```bash
python llm.py
```

The script will:
- Process three sample patient cases
- Display diagnosis probabilities for each case
- Analyze attention patterns across multiple attention heads
- Generate attention visualization heatmaps
- Save visualization as `medical_diagnosis_attention.png`

## Sample Output

For each patient case, the system outputs:
- **Presenting Symptoms**: List of reported symptoms
- **Vital Signs**: Temperature, heart rate, blood pressure, oxygen saturation, etc.
- **Diagnosis Probabilities**: Predicted probability for each disease
- **Primary Diagnosis**: Highest probability diagnosis
- **Attention Analysis**: What each attention head focuses on

## Example Patient Cases

### Patient A - Suspected Flu
- Symptoms: Fever, cough, body ache, fatigue, headache
- Vitals: Temperature 101.5°F, Heart rate 95, O2 saturation 97%
- Expected: Influenza

### Patient B - Respiratory Issues
- Symptoms: Fever, cough, shortness of breath, chest pain, fatigue
- Vitals: Temperature 102.8°F, Heart rate 105, O2 saturation 92%, Respiratory rate 24
- Expected: Pneumonia

### Patient C - Cardiac Symptoms
- Symptoms: Chest pain, shortness of breath, dizziness, sweating
- Vitals: Heart rate 115, Blood pressure 160/95
- Expected: Heart Disease

## Supported Diseases

- Influenza
- Pneumonia
- COVID-19
- Heart Disease
- Healthy

## Supported Symptoms

Fever, cough, fatigue, chest pain, shortness of breath, headache, body ache, sore throat, runny nose, nausea, dizziness, rapid heartbeat, sweating

## Supported Vital Signs

Temperature, heart rate, blood pressure (systolic/diastolic), oxygen saturation, respiratory rate

## Configuration

You can modify the system configuration in `llm.py`:

```python
d_model = 32          # Embedding dimension
num_heads = 4         # Number of attention heads
```

## How Attention Works in Medical Diagnosis

Each attention head specializes in different diagnostic patterns:

- **Head 1**: May focus on respiratory symptoms (cough + oxygen saturation)
- **Head 2**: May focus on cardiovascular signs (chest pain + heart rate)
- **Head 3**: May focus on inflammatory markers (fever + body ache)
- **Head 4**: May focus on vital sign correlations

This multi-perspective analysis leads to more robust and interpretable diagnoses.

## Visualization

The script generates `medical_diagnosis_attention.png` showing:
- Attention heatmaps for each attention head
- Patient case information and predicted diagnosis
- Feature importance patterns

## Important Disclaimer

⚠️ **This is a demonstration system for educational purposes only.**

- NOT intended for actual medical diagnosis
- Always requires human physician oversight
- Attention weights show correlation, not causation
- Must be trained on validated medical datasets
- Requires regulatory approval before clinical use
- Should augment, not replace, clinical judgment

## Benefits of Attention Mechanisms in Healthcare

1. **Interpretability**: Shows which symptoms/vitals drive diagnoses
2. **Feature Relationships**: Discovers symptom interactions automatically
3. **Robust Handling**: Works with variable numbers of symptoms/vitals
4. **Multiple Perspectives**: Different diagnostic reasoning paths
5. **Transparency**: Provides explainability for medical decision support

## Real-World Applications

- Emergency triage systems
- Diagnostic support systems
- Patient monitoring in ICU
- Drug interaction analysis
- Genomic medicine
- Telemedicine platforms

## Technical Details

- **Model Dimension**: 32 (embedding size)
- **Attention Heads**: 4
- **Classification**: Softmax over disease categories
- **Pooling**: Combined max and mean pooling of attention outputs
- **Visualization**: Matplotlib heatmaps

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Visualization and heatmap generation

## Files

- `llm.py` - Main implementation
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `medical_diagnosis_attention.png` - Generated visualization

## Author Notes

This implementation demonstrates how pure Python functions can implement complex deep learning concepts like multi-head attention without requiring frameworks like PyTorch or TensorFlow. All computations use NumPy for transparency and educational clarity.

## License

Educational demonstration project - free to modify and distribute

## Contact & Questions

For questions about the implementation or attention mechanisms in healthcare AI, refer to the comments and docstrings throughout `llm.py`.
