import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from llm import (
    softmax, scaled_dot_product_attention, self_attention_forward,
    split_heads, concat_heads, multi_head_attention_forward,
    initialize_encoders, encode_symptom, encode_vital, encode_patient_data,
    initialize_diagnosis_system, diagnose,
    SYMPTOMS, VITALS, DISEASES, PATIENT_CASES
)

# Page config
st.set_page_config(
    page_title="Medical Diagnosis System - Attention Mechanisms",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 1.8em;
        color: #2ca02c;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 10px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("# 🏥 Navigation")
page = st.sidebar.radio(
    "Select a Section:",
    ["Home", "Attention Mechanisms Demo", "Medical Diagnosis System", "How It Works"]
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    st.markdown("# 🏥 Medical Diagnosis System Using Attention Mechanisms")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to an interactive demonstration of **Multi-Head Attention** applied to 
        medical diagnosis! This system analyzes patient symptoms and vital signs to 
        predict potential diseases while showing which factors influenced each diagnosis.
        """)
    
    with col2:
        st.info("👈 Use the sidebar to navigate between sections")
    
    # Key features
    st.markdown("## ✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 Attention Mechanisms
        - Self-Attention
        - Multi-Head Attention
        - Interpretable weights
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Medical Analysis
        - Symptom encoding
        - Vital sign analysis
        - Disease prediction
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Visualization
        - Attention heatmaps
        - Probability charts
        - Feature importance
        """)
    
    # System specs
    st.markdown("## 🔧 System Configuration")
    spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
    
    with spec_col1:
        st.metric("Model Dimension", "32")
    with spec_col2:
        st.metric("Attention Heads", "4")
    with spec_col3:
        st.metric("Diseases Tracked", len(DISEASES))
    with spec_col4:
        st.metric("Symptoms", len(SYMPTOMS))

# ============================================================================
# ATTENTION MECHANISMS DEMO
# ============================================================================
elif page == "Attention Mechanisms Demo":
    st.markdown("# 🧠 Attention Mechanisms Demonstration")
    
    st.markdown("""
    This section shows how Self-Attention and Multi-Head Attention work with concrete numerical examples.
    """)
    
    # Create sample input
    np.random.seed(42)
    demo_input = np.random.randn(4, 8)
    d_model_demo = 8
    
    # Initialize weights
    W_q_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
    W_k_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
    W_v_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
    W_o_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
    
    # ---- STEP 1: INPUT ----
    st.markdown("## Step 1️⃣ : Input Data")
    st.write("""
    We start with a sequence of 4 tokens, each with 8 dimensions. This could represent 
    4 patient features (like symptoms or vital signs) encoded as 8-dimensional vectors.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Input Shape:**")
        st.code(f"(4, 8)\n4 tokens × 8 dimensions", language="text")
    with col2:
        st.write("**Input Data:**")
        input_df = np.round(demo_input, 3)
        st.dataframe(input_df, use_container_width=True)
    
    # ---- STEP 2: SELF-ATTENTION ----
    st.markdown("## Step 2️⃣ : Self-Attention Mechanism")
    
    st.write("""
    Self-Attention computes how much each token should attend to every other token:
    1. **Query (Q)**: What am I looking for?
    2. **Key (K)**: What information do I have?
    3. **Value (V)**: What should I pass on?
    
    The attention score is computed as: **Attention = softmax(QK^T / √d) * V**
    """)
    
    self_attn_output, self_attn_weights = self_attention_forward(
        demo_input, W_q_demo, W_k_demo, W_v_demo, d_model_demo
    )
    
    tab1, tab2 = st.tabs(["Attention Weights", "Output"])
    
    with tab1:
        st.write("**Attention Weight Matrix** - Shows how much each token attends to others:")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            self_attn_weights,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=['Token 0', 'Token 1', 'Token 2', 'Token 3'],
            yticklabels=['Token 0', 'Token 1', 'Token 2', 'Token 3']
        )
        ax.set_xlabel('Attending To (Key)')
        ax.set_ylabel('From (Query)')
        ax.set_title('Self-Attention Weights')
        st.pyplot(fig)
        
        st.info("""
        💡 **Interpretation**: Each row sums to 1.0 (softmax). 
        For example, Token 0 might attend 96.7% to Token 3 and ignore others.
        """)
    
    with tab2:
        st.write("**Self-Attention Output** - Transformed tokens after attention:")
        output_df = np.round(self_attn_output, 3)
        st.dataframe(output_df, use_container_width=True)
    
    # ---- STEP 3: MULTI-HEAD ATTENTION ----
    st.markdown("## Step 3️⃣ : Multi-Head Attention (2 Heads)")
    
    st.write("""
    Multi-Head Attention runs multiple attention mechanisms in parallel:
    - **Head 1** might focus on one type of relationship
    - **Head 2** might focus on a different type of relationship
    - Results are concatenated and projected
    
    This allows the model to capture multiple patterns simultaneously.
    """)
    
    num_heads_demo = 2
    multi_head_output, multi_head_weights = multi_head_attention_forward(
        demo_input, W_q_demo, W_k_demo, W_v_demo, W_o_demo,
        d_model_demo, num_heads_demo
    )
    
    # Display numerical weights first
    st.markdown("## Multi-Head Attention Weights (Numerical)")
    
    for head_idx in range(num_heads_demo):
        st.write(f"### Head {head_idx + 1} Attention Weights:")
        st.code(str(np.round(multi_head_weights[head_idx], 3)), language="text")
    
    st.write("### Combined Output (All Heads):")
    st.code(str(np.round(multi_head_output, 3)), language="text")
    
    st.markdown("---")
    st.markdown("## Multi-Head Attention Visualization")
    
    # Display each head
    for head_idx in range(num_heads_demo):
        st.subheader(f"Head {head_idx + 1}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Attention Weights**")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                multi_head_weights[head_idx],
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                ax=ax,
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=['T0', 'T1', 'T2', 'T3'],
                yticklabels=['T0', 'T1', 'T2', 'T3']
            )
            ax.set_title(f'Head {head_idx + 1} Attention')
            st.pyplot(fig)
        
        with col2:
            st.write("**Key Observations**")
            avg_attn = np.mean(multi_head_weights[head_idx], axis=0)
            top_attended = np.argsort(avg_attn)[::-1][:2]
            
            for rank, token_idx in enumerate(top_attended, 1):
                st.write(f"{rank}. Token {token_idx}: {avg_attn[token_idx]:.2%} average attention")

# ============================================================================
# MEDICAL DIAGNOSIS SYSTEM
# ============================================================================
elif page == "Medical Diagnosis System":
    st.markdown("# 🏥 Medical Diagnosis System")
    
    # Initialize system
    d_model = 32
    num_heads = 4
    diagnosis_system = initialize_diagnosis_system(d_model, num_heads, len(DISEASES))
    
    st.markdown("""
    Select a patient case or create a custom case to see how the system 
    analyzes symptoms and vital signs to predict diseases.
    """)
    
    # Patient selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        patient_choice = st.selectbox(
            "Select Patient Case:",
            options=[f"Case {i+1}: {p['name']}" for i, p in enumerate(PATIENT_CASES)],
            key="patient_select"
        )
    
    with col2:
        case_idx = int(patient_choice.split(":")[0].split()[-1]) - 1
    
    patient = PATIENT_CASES[case_idx]
    
    # Display patient information
    st.markdown("## 📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Symptoms")
        for symptom in patient['symptoms']:
            st.write(f"• {symptom.replace('_', ' ').title()}")
    
    with col2:
        st.subheader("Vital Signs")
        vitals_text = ""
        for vital_name, vital_value in patient['vitals']:
            st.write(f"• {vital_name.replace('_', ' ').title()}: {vital_value}")
    
    # Run diagnosis
    symptom_ids = [SYMPTOMS[s] for s in patient['symptoms']]
    vital_data = [(VITALS[v[0]], v[1]) for v in patient['vitals']]
    feature_names = patient['symptoms'] + [f"{v[0]}={v[1]}" for v in patient['vitals']]
    
    probs, attn_output, attn_weights, features = diagnose(
        symptom_ids, vital_data, diagnosis_system, feature_names
    )
    
    # ---- DIAGNOSIS RESULTS ----
    st.markdown("## 🔍 Diagnosis Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Probability chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_indices = np.argsort(probs)[::-1]
        sorted_diseases = [DISEASES[i] for i in sorted_indices]
        sorted_probs = probs[sorted_indices]
        
        colors = ['#d62728' if disease == patient['expected'] else '#1f77b4' 
                 for disease in sorted_diseases]
        
        bars = ax.barh(sorted_diseases, sorted_probs, color=colors)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title('Disease Prediction Probabilities', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(sorted_probs) * 1.1)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
            ax.text(prob + 0.01, i, f'{prob:.1%}', va='center', fontsize=10)
        
        st.pyplot(fig)
    
    with col2:
        st.write("### Prediction Summary")
        predicted_disease = DISEASES[np.argmax(probs)]
        predicted_prob = probs[np.argmax(probs)]
        
        if predicted_disease == patient['expected']:
            st.success(f"✅ **Correct!**\n\n**Predicted**: {predicted_disease}\n\n**Confidence**: {predicted_prob:.1%}")
        else:
            st.warning(f"❌ **Incorrect**\n\n**Predicted**: {predicted_disease}\n\n**Expected**: {patient['expected']}")
    
    # ---- ATTENTION ANALYSIS ----
    st.markdown("## 🧠 Attention Analysis")
    
    st.write("""
    Different attention heads focus on different patterns. Here's what each head prioritizes:
    """)
    
    for head_idx in range(num_heads):
        with st.expander(f"Head {head_idx + 1} Analysis", expanded=(head_idx == 0)):
            avg_attention = np.mean(attn_weights[head_idx], axis=0)
            top_indices = np.argsort(avg_attention)[-5:][::-1]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Top 5 Features:**")
                for rank, idx in enumerate(top_indices, 1):
                    if idx < len(features):
                        feature = features[idx]
                        score = avg_attention[idx]
                        st.write(f"{rank}. {feature}\n   Score: {score:.3f}")
            
            with col2:
                # Attention heatmap for this head
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    attn_weights[head_idx],
                    annot=False,
                    cmap='YlOrRd',
                    ax=ax,
                    cbar_kws={'label': 'Attention Weight'},
                    xticklabels=features,
                    yticklabels=features
                )
                ax.set_title(f'Head {head_idx + 1} Attention Pattern', fontweight='bold')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
                st.pyplot(fig)
    
    # ---- COMPREHENSIVE HEATMAP ----
    st.markdown("## 📊 Multi-Head Attention Visualization")
    
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 6))
    if num_heads == 1:
        axes = [axes]
    
    for head_idx in range(num_heads):
        sns.heatmap(
            attn_weights[head_idx],
            annot=False,
            cmap='RdYlBu_r',
            ax=axes[head_idx],
            cbar_kws={'label': 'Attention Weight'},
            vmin=0, vmax=1,
            xticklabels=features,
            yticklabels=features
        )
        axes[head_idx].set_title(f'Head {head_idx + 1}', fontweight='bold', fontsize=12)
        plt.setp(axes[head_idx].get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(axes[head_idx].get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# HOW IT WORKS
# ============================================================================
elif page == "How It Works":
    st.markdown("# 📚 How It Works - Step by Step")
    
    st.markdown("""
    This guide explains how the Medical Diagnosis System using Attention Mechanisms works.
    """)
    
    # Step 1
    with st.expander("Step 1: Feature Encoding", expanded=True):
        st.markdown("""
        Convert symptoms & vital signs into numbers (embeddings):
        - Fever → `[0.1, -0.2, 0.05, ...]`
        - Each position captures different aspects
        - Vital signs are normalized so 99°F ≠ 101.5°F
        """)
    
    # Step 2
    with st.expander("Step 2: Self-Attention"):
        st.markdown("""
        Each feature learns which other features matter most.
        
        **Three steps:**
        1. **Query (Q)**: What am I looking for?
        2. **Key (K)**: What info is available?
        3. **Value (V)**: What should I use?
        
        **Formula**: `Attention = softmax(QK^T / √d) × V`
        
        **Example**: If fever + low O₂ = serious → they learn to pay attention to each other.
        """)
    
    # Step 3
    with st.expander("Step 3: Multiple Attention Heads"):
        st.markdown("""
        Use 4 heads to capture different patterns:
        - **Head 1**: Respiratory (cough ↔ O₂ saturation)
        - **Head 2**: Cardiovascular (chest pain ↔ heart rate)
        - **Head 3**: Inflammation (fever ↔ body ache)
        - **Head 4**: General correlations
        
        Split 32D → 4 heads × 8D each → Apply attention → Concatenate back
        """)
    
    # Step 4
    with st.expander("Step 4: Make Prediction"):
        st.markdown("""
        1. **Combine all features**: Average max & mean of all features
        2. **Classify**: Map to disease probabilities
        3. **Predict**: Pick disease with highest probability
        
        **Output**: Pneumonia 35% → COVID-19 20% → Flu 25% → ...
        """)
    
    # Step 5
    with st.expander("Step 5: Understand Attention Weights"):
        st.markdown("""
        Heatmap shows what model "paid attention to":
        
        If fever → O₂ saturation = 0.85:
        - Fever noticed O₂ saturation 85% of the time
        - Model learned: Low O₂ + fever = more serious
        
        **Why it matters:**
        - See what influenced diagnosis ✓
        - Doctors can verify decisions ✓
        - Debug wrong predictions ✓
        """)
    
    # Key Concepts
    st.markdown("## 🎓 Quick Summary")
    
    st.markdown("""
    - **Embedding**: Convert words → numbers
    - **Attention**: Learn which features matter
    - **Multi-Head**: Capture multiple patterns at once
    - **Diagnosis**: Combine everything to predict disease
    """)
    
    st.markdown("## ✅ Why This Works")
    st.markdown("""
    - **Clear**: You can see what influenced decision
    - **Flexible**: Handles any number of symptoms
    - **Smart**: Finds patterns doctors might miss
    - **Fast**: Uses simple math (NumPy)
    """)
    
    st.markdown("## ⚠️ Important")
    st.warning("❌ NOT for real medical use • ⚠️ Consult doctors first")

st.markdown("---")
st.markdown("""
<div style='text-align: center; margin-top: 30px; color: #666;'>
<p>🏥 Medical Diagnosis System using Self-Attention and Multi-Head Attention</p>
<p style='font-size: 0.9em;'>Educational Demonstration • Not for Medical Use</p>
</div>
""", unsafe_allow_html=True)
