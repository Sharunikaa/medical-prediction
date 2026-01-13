"""
Simple Medical Diagnosis System with Attention Mechanisms
Uses self-attention and multi-head attention for interpretable predictions
Optional: Gemini API integration for AI explanations
"""

import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# GEMINI API CONFIGURATION
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================================
# 1. CORE FUNCTIONS
# ============================================================================

def softmax(x):
    """Convert to probabilities (0-1)"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, d_model):
    """
    Attention = softmax(Q @ K.T / sqrt(d)) @ V
    
    Q: Query (what to look for)
    K: Key (what's available)
    V: Value (what to combine)
    """
    scores = np.dot(Q, K.T) / np.sqrt(d_model)
    weights = softmax(scores)
    output = np.dot(weights, V)
    return output, weights


def self_attention(X, W_q, W_k, W_v, d_model):
    """Apply self-attention to input X"""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    output, weights = scaled_dot_product_attention(Q, K, V, d_model)
    return weights, output


def multi_head_attention(X, WQ, WK, WV, WO, num_heads, d_model):
    """Apply multi-head attention"""
    d_k = d_model // num_heads
    head_outputs = []
    all_weights = []
    
    for i in range(num_heads):
        Q = np.dot(X, WQ[i])
        K = np.dot(X, WK[i])
        V = np.dot(X, WV[i])
        weights, output = scaled_dot_product_attention(Q, K, V, d_k)
        head_outputs.append(output)
        all_weights.append(weights)
    
    concatenated = np.concatenate(head_outputs, axis=-1)
    final_output = np.dot(concatenated, WO)
    
    return all_weights, final_output


# ============================================================================
# 2. MEDICAL SYSTEM
# ============================================================================

# Define medical features
SYMPTOMS = {
    'fever': 0, 'cough': 1, 'fatigue': 2, 'chest_pain': 3,
    'shortness_of_breath': 4, 'headache': 5, 'body_ache': 6
}

VITALS = {
    'temperature': 0, 'heart_rate': 1, 'oxygen_saturation': 2
}

DISEASES = ['Influenza', 'Pneumonia', 'COVID-19', 'Heart Disease', 'Healthy']


def initialize_system(d_model=32, num_diseases=5):
    """Create neural network weights"""
    return {
        'W_q': np.random.randn(d_model, d_model) * 0.1,
        'W_k': np.random.randn(d_model, d_model) * 0.1,
        'W_v': np.random.randn(d_model, d_model) * 0.1,
        'W_o': np.random.randn(d_model, d_model) * 0.1,
        'W_class': np.random.randn(d_model, num_diseases) * 0.1,
        'W_symptom': np.random.randn(len(SYMPTOMS), d_model) * 0.1,
        'W_vital': np.random.randn(len(VITALS), d_model) * 0.1,
        'bias': np.zeros(num_diseases),
        'd_model': d_model
    }


def encode_features(symptoms_list, vitals_list, system):
    """Convert symptoms & vitals to embeddings"""
    d_model = system['d_model']
    
    # Encode symptoms (one-hot)
    symptom_encodings = []
    for symptom in symptoms_list:
        idx = SYMPTOMS[symptom]
        encoding = system['W_symptom'][idx]
        symptom_encodings.append(encoding)
    
    # Encode vitals (value + embedding)
    vital_encodings = []
    for vital_name, vital_value in vitals_list:
        idx = VITALS[vital_name]
        base_encoding = system['W_vital'][idx]
        normalized = np.tanh(vital_value / 100.0)
        encoding = base_encoding * (1 + normalized)
        vital_encodings.append(encoding)
    
    # Combine all
    all_encodings = np.array(symptom_encodings + vital_encodings)
    return all_encodings


def diagnose(symptoms_list, vitals_list, system):
    """Make diagnosis"""
    # Step 1: Encode features
    X = encode_features(symptoms_list, vitals_list, system)
    
    # Step 2: Apply attention
    attn_weights, attn_output = self_attention(
        X, system['W_q'], system['W_k'], system['W_v'], system['d_model']
    )
    
    # Step 3: Pool features
    pooled = (np.max(attn_output, axis=0) + np.mean(attn_output, axis=0)) / 2
    
    # Step 4: Classify
    logits = np.dot(pooled, system['W_class']) + system['bias']
    probs = softmax(logits)
    
    return probs, attn_weights, X, attn_output


if __name__ == "__main__":
    print("=" * 70)
    print("MEDICAL DIAGNOSIS SYSTEM WITH ATTENTION MECHANISMS")
    print("=" * 70)
    
    # Initialize system
    system = initialize_system(d_model=16, num_diseases=len(DISEASES))
    
    # ====================================================================
    # PATIENT 1: SUSPECTED FLU
    # ====================================================================
    print("\n📋 PATIENT 1: Fever + Cough + Fatigue")
    print("-" * 70)
    
    symptoms_1 = ['fever', 'cough', 'fatigue', 'body_ache']
    vitals_1 = [
        ('temperature', 101.5),
        ('heart_rate', 95),
        ('oxygen_saturation', 97)
    ]
    
    probs_1, attn_1, X_1, attn_output_1 = diagnose(symptoms_1, vitals_1, system)
    
    print(f"Symptoms: {', '.join(symptoms_1)}")
    print(f"Vitals: Temp=101.5°F, HR=95, O₂=97%")
    print(f"\n🔍 PREDICTIONS:")
    for i, disease in enumerate(DISEASES):
        prob_percent = probs_1[i] * 100
        bar = "█" * int(prob_percent / 5)
        print(f"  {disease:15} {prob_percent:5.1f}% {bar}")
    
    predicted_1 = DISEASES[np.argmax(probs_1)]
    confidence_1 = np.max(probs_1)
    print(f"\n✓ Top prediction: {predicted_1} ({confidence_1:.1%})")
    
    print(f"\n🧠 SELF-ATTENTION WEIGHTS:")
    print("(How much each symptom/vital attends to others)")
    print(attn_1)
    
    print(f"\n💡 Key observations:")
    avg_attention = np.mean(attn_1, axis=0)
    top_indices = np.argsort(avg_attention)[-3:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        feature_name = list(SYMPTOMS.keys())[idx] if idx < len(SYMPTOMS) else "vital_sign"
        score = avg_attention[idx]
        print(f"  {rank}. {feature_name:20} {score:.3f}")
    
    # ====================================================================
    # PATIENT 2: RESPIRATORY ISSUES
    # ====================================================================
    print("\n" + "=" * 70)
    print("\n📋 PATIENT 2: High Fever + Low O₂ (Respiratory)")
    print("-" * 70)
    
    symptoms_2 = ['fever', 'cough', 'shortness_of_breath', 'chest_pain']
    vitals_2 = [
        ('temperature', 102.8),
        ('heart_rate', 105),
        ('oxygen_saturation', 92)  # LOW!
    ]
    
    probs_2, attn_2, X_2, attn_output_2 = diagnose(symptoms_2, vitals_2, system)
    
    print(f"Symptoms: {', '.join(symptoms_2)}")
    print(f"Vitals: Temp=102.8°F, HR=105, O₂=92% ⚠️ LOW")
    print(f"\n🔍 PREDICTIONS:")
    for i, disease in enumerate(DISEASES):
        prob_percent = probs_2[i] * 100
        bar = "█" * int(prob_percent / 5)
        print(f"  {disease:15} {prob_percent:5.1f}% {bar}")
    
    predicted_2 = DISEASES[np.argmax(probs_2)]
    confidence_2 = np.max(probs_2)
    print(f"\n✓ Top prediction: {predicted_2} ({confidence_2:.1%})")
    
    print(f"\n🧠 SELF-ATTENTION WEIGHTS:")
    print(attn_2)
    
    print(f"\n💡 Key observations:")
    avg_attention_2 = np.mean(attn_2, axis=0)
    top_indices_2 = np.argsort(avg_attention_2)[-3:][::-1]
    for rank, idx in enumerate(top_indices_2, 1):
        feature_name = list(SYMPTOMS.keys())[idx] if idx < len(SYMPTOMS) else "vital_sign"
        score = avg_attention_2[idx]
        print(f"  {rank}. {feature_name:20} {score:.3f}")
    
    # ====================================================================
    # GEMINI API EXPLANATION (OPTIONAL - requires Google API key)
    # ====================================================================
    prompt = f"""Analyze these medical diagnosis results using attention mechanisms:

PATIENT 1: Suspected Flu
- Symptoms: {', '.join(symptoms_1)}
- Vitals: Temp=101.5°F, HR=95, O₂=97%
- Predicted Disease: {predicted_1}
- Confidence: {confidence_1:.1%}

PATIENT 2: Respiratory Issues
- Symptoms: {', '.join(symptoms_2)}
- Vitals: Temp=102.8°F, HR=105, O₂=92%
- Predicted Disease: {predicted_2}
- Confidence: {confidence_2:.1%}

Self-Attention Weights for Patient 1 (rows=symptoms, cols=symptoms):
{np.round(attn_1, 3)}

Self-Attention Weights for Patient 2:
{np.round(attn_2, 3)}

Provide concise explanation:
1. How attention mechanisms identify important symptom relationships
2. Why Patient 2's low oxygen level creates different attention patterns
3. What clinical insights the attention weights reveal
4. How this improves diagnosis interpretability"""

    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            print("\n" + "=" * 70)
            print("QUERYING GEMINI API FOR AI INTERPRETATION...")
            print("=" * 70)
            
            response = model.generate_content(prompt)
            
            print("\n" + "=" * 70)
            print("GEMINI AI ANALYSIS")
            print("=" * 70)
            print(response.text)
            
        except Exception as e:
            print("\n" + "=" * 70)
            print("NOTE: Could not connect to Gemini API")
            print("=" * 70)
            print(f"Error: {e}")
    else:
        print("\n" + "=" * 70)
        print("NOTE: Gemini API not available")
        print("=" * 70)
        if not GEMINI_API_KEY:
            print("Missing GEMINI_API_KEY in .env file")
            print("Get key: https://makersuite.google.com/app/apikey")
        if not GEMINI_AVAILABLE:
            print("google-generativeai not installed")
            print("Install: pip install google-generativeai python-dotenv")
    
    # ====================================================================
    # MULTI-HEAD ATTENTION DEMO (Separate)
    # ====================================================================
    print("\n" + "=" * 70)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("=" * 70)
    
    # Create simple demo input for multi-head
    np.random.seed(42)
    demo_mha_input = np.random.randn(4, 8)  # 4 features, 8 dimensions
    d_model_demo = 8
    num_heads_demo = 2
    d_k_demo = d_model_demo // num_heads_demo  # 4
    
    print(f"\nDemo Input shape: {demo_mha_input.shape}")
    print(f"Model dimension: {d_model_demo}")
    print(f"Number of heads: {num_heads_demo}")
    print(f"Dimension per head: {d_k_demo}")
    
    # Initialize weights for demo
    WQ_demo = [np.random.randn(d_model_demo, d_k_demo) * 0.1 for _ in range(num_heads_demo)]
    WK_demo = [np.random.randn(d_model_demo, d_k_demo) * 0.1 for _ in range(num_heads_demo)]
    WV_demo = [np.random.randn(d_model_demo, d_k_demo) * 0.1 for _ in range(num_heads_demo)]
    WO_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
    
    mh_weights_demo, mh_output_demo = multi_head_attention(
        demo_mha_input, WQ_demo, WK_demo, WV_demo, WO_demo, num_heads_demo, d_model_demo
    )
    
    for head_idx, weights in enumerate(mh_weights_demo):
        print(f"\n📊 HEAD {head_idx + 1} ATTENTION WEIGHTS:")
        print(np.round(weights, 3))
    
    print(f"\n📊 ALL HEADS COMBINED OUTPUT:")
    print(np.round(mh_output_demo, 3))
    print("\n" + "=" * 70)
    print("HOW IT WORKS")
    print("=" * 70)
    print("""
1️⃣  ENCODING
    Convert symptoms & vitals → embeddings (vectors)

2️⃣  SELF-ATTENTION
    Learn which features relate to each other
    High weight = strong relationship

3️⃣  POOLING
    Combine all attention outputs

4️⃣  CLASSIFICATION
    Map to disease probabilities (0-100%)

5️⃣  PREDICTION
    Pick highest probability disease

Why Attention Matters:
✓ TRANSPARENT - See which symptoms influenced diagnosis
✓ FLEXIBLE - Handle any combination of symptoms
✓ SMART - Learn complex symptom relationships
✓ EXPLAINABLE - Attention weights explain decisions
    """)
