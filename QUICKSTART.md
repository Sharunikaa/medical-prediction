# 🚀 Quick Start Guide - Streamlit Frontend

## Installation

1. Navigate to the project directory:
```bash
cd "/Users/Sharunikaa/LLM lab/self_attention"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Execute the following command:
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Features

### 📍 Home Page
- Overview of the system
- Key features at a glance
- System configuration metrics

### 🧠 Attention Mechanisms Demo
**Step-by-step walkthrough of how attention works:**

1. **Input Data** - Shows the initial 4×8 sample data
2. **Self-Attention** - Demonstrates single-head attention
   - Visualizes attention weight matrix
   - Shows how tokens attend to each other
   - Displays transformed output
3. **Multi-Head Attention** - Shows parallel attention heads
   - Individual heatmaps for each head
   - Key observations per head

### 🏥 Medical Diagnosis System
**Interactive diagnosis tool:**

1. **Patient Selection** - Choose from 3 sample cases or see custom inputs
2. **Patient Information** - Displays symptoms and vital signs
3. **Diagnosis Results** - Shows:
   - Probability bar chart for all 5 diseases
   - Predicted diagnosis vs expected diagnosis
   - Confidence score
4. **Attention Analysis** - Deep dive into what each head focuses on
   - Top 5 features per head
   - Attention heatmap for each head
5. **Multi-Head Visualization** - Combined view of all 4 heads

### 📚 How It Works
**Educational section with step-by-step explanations:**

- Step 1: Symptom & Vital Sign Encoding
- Step 2: Self-Attention Mechanism
- Step 3: Multi-Head Attention
- Step 4: Classification & Diagnosis
- Step 5: Attention Weight Interpretation
- Key Concepts Summary
- Advantages and Limitations

## Sample Patient Cases

### Patient A - Suspected Flu
- **Symptoms**: Fever, Cough, Body Ache, Fatigue, Headache
- **Vitals**: Temperature 101.5°F, HR 95, O2 97%
- **Expected**: Influenza

### Patient B - Respiratory Issues
- **Symptoms**: Fever, Cough, Shortness of Breath, Chest Pain, Fatigue
- **Vitals**: Temperature 102.8°F, HR 105, O2 92%, RR 24
- **Expected**: Pneumonia

### Patient C - Cardiac Symptoms
- **Symptoms**: Chest Pain, Shortness of Breath, Dizziness, Sweating
- **Vitals**: HR 115, BP 160/95
- **Expected**: Heart Disease

## Interactive Elements

### Expander Sections
- Click to expand/collapse detailed information
- Starts with first section expanded for easy navigation

### Tabs
- Switch between different views (e.g., Attention Weights vs Output)

### Charts & Visualizations
- Probability bars with color coding
- Heatmaps showing attention patterns
- Support for interactive exploration

## Tips for Best Experience

1. **Full Screen Mode** - Use browser full screen (F11) for better visualization of heatmaps
2. **Sidebar Navigation** - Always visible, easy to switch between sections
3. **Responsive Design** - Works on desktop and tablet screens
4. **Dark Mode** - Streamlit supports dark theme (Settings → Theme)

## Understanding the Visualizations

### Heatmaps
- **Brighter colors** = Higher attention weight
- **Rows** = Features attending (from)
- **Columns** = Features being attended to (to)
- **Values** = Probability (0.0 to 1.0)

### Probability Charts
- **Red bar** = Expected diagnosis
- **Blue bars** = Other diseases
- **Numbers** = Percentage probability

## Troubleshooting

### Streamlit not starting?
```bash
# Make sure you're in the right directory
pwd

# Reinstall streamlit
pip install --upgrade streamlit
```

### Port already in use?
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

### Memory issues with visualizations?
- Reduce the number of open tabs/sections
- The visualizations are generated on-demand

## File Structure

```
self_attention/
├── llm.py                 # Main model implementation
├── streamlit_app.py       # Streamlit frontend
├── requirements.txt       # Python dependencies
├── README.md             # Full documentation
└── QUICKSTART.md         # This file
```

## Next Steps

1. **Explore the Demo** - Start with "Attention Mechanisms Demo"
2. **Try Different Cases** - Use "Medical Diagnosis System" with different patients
3. **Learn the Concepts** - Read through "How It Works" section
4. **Understand Attention** - Check out attention weight heatmaps

## Educational Value

This system demonstrates:
- ✅ How attention mechanisms work in detail
- ✅ Multi-head attention benefits
- ✅ Real-world medical AI applications
- ✅ Explainable AI through attention weights
- ✅ NumPy-based implementation (no PyTorch/TF)

## Important Disclaimer

⚠️ **This is an educational demonstration, NOT for real medical use**
- Always consult qualified healthcare professionals
- Requires proper training data and validation
- Needs regulatory approval before clinical use
- Use as a learning tool only

Enjoy exploring the Medical Diagnosis System! 🏥
