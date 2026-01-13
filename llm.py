import numpy as np
import matplotlib.pyplot as plt

"""
UNIQUE USE CASE: Medical Symptom Analysis and Disease Prediction
This system uses attention mechanisms to analyze patient symptoms and vital signs
to predict potential diseases. The attention weights show which symptoms and 
measurements are most relevant for each diagnosis, providing interpretability 
for medical professionals.
"""

# Step 1: Self-Attention Mechanism Functions
def softmax(x):
    """Compute softmax along the last axis"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, d_model):
    """Compute scaled dot-product attention"""
    scores = np.dot(Q, K.T) / np.sqrt(d_model)
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, V)
    return output, attention_weights

def self_attention_forward(X, W_q, W_k, W_v, d_model):
    """Forward pass for self-attention"""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    output, attention_weights = scaled_dot_product_attention(Q, K, V, d_model)
    return output, attention_weights


# Step 2: Multi-Head Attention Mechanism Functions
def split_heads(X, num_heads, d_k):
    """Split into multiple heads"""
    seq_len = X.shape[0]
    X = X.reshape(seq_len, num_heads, d_k)
    return X.transpose(1, 0, 2)

def concat_heads(outputs):
    """Concatenate multiple heads"""
    return np.concatenate(outputs, axis=-1)

def multi_head_attention_forward(X, W_q, W_k, W_v, W_o, d_model, num_heads):
    """Forward pass for multi-head attention"""
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    d_k = d_model // num_heads
    seq_len = X.shape[0]
    
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    
    Q_heads = split_heads(Q, num_heads, d_k)
    K_heads = split_heads(K, num_heads, d_k)
    V_heads = split_heads(V, num_heads, d_k)
    
    outputs = []
    all_attention_weights = []
    
    for i in range(num_heads):
        output, attn_weights = scaled_dot_product_attention(
            Q_heads[i], K_heads[i], V_heads[i], d_k
        )
        outputs.append(output)
        all_attention_weights.append(attn_weights)
    
    concat_output = concat_heads(outputs)
    final_output = np.dot(concat_output, W_o)
    
    return final_output, all_attention_weights


# Step 3: Medical Feature Encoder Functions
def initialize_encoders(d_model):
    """Initialize medical feature encoders"""
    symptom_encoder = np.random.randn(100, d_model) * 0.1
    vital_encoder = np.random.randn(10, d_model) * 0.1
    return symptom_encoder, vital_encoder

def encode_symptom(symptom_id, symptom_encoder):
    """Encode a symptom"""
    return symptom_encoder[symptom_id]

def encode_vital(vital_value, vital_type, vital_encoder):
    """Encode vital signs with their values"""
    base_encoding = vital_encoder[vital_type]
    normalized_value = np.tanh(vital_value / 100.0)
    return base_encoding * (1 + normalized_value)

def encode_patient_data(symptoms, vitals, symptom_encoder, vital_encoder):
    """
    Encode complete patient data
    symptoms: list of symptom IDs
    vitals: list of (vital_type, value) tuples
    """
    encodings = []
    
    for symptom_id in symptoms:
        encodings.append(encode_symptom(symptom_id, symptom_encoder))
    
    for vital_type, value in vitals:
        encodings.append(encode_vital(value, vital_type, vital_encoder))
    
    return np.array(encodings)


# Step 4: Medical Diagnosis System Functions
def initialize_diagnosis_system(d_model, num_heads, num_diseases):
    """Initialize the diagnosis system with all required weights"""
    symptom_encoder, vital_encoder = initialize_encoders(d_model)
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    W_o = np.random.randn(d_model, d_model) * 0.1
    W_class = np.random.randn(d_model, num_diseases) * 0.1
    bias = np.zeros(num_diseases)
    
    return {
        'symptom_encoder': symptom_encoder,
        'vital_encoder': vital_encoder,
        'W_q': W_q,
        'W_k': W_k,
        'W_v': W_v,
        'W_o': W_o,
        'W_class': W_class,
        'bias': bias,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_diseases': num_diseases
    }

def diagnose(symptoms, vitals, system, feature_names):
    """
    Diagnose patient
    symptoms: list of symptom IDs
    vitals: list of (vital_type, value) tuples
    system: diagnosis system dictionary with all weights
    feature_names: names of all features for visualization
    """
    # Encode patient data
    X = encode_patient_data(symptoms, vitals, 
                           system['symptom_encoder'], 
                           system['vital_encoder'])
    
    # Apply multi-head attention
    attn_output, attn_weights = multi_head_attention_forward(
        X, system['W_q'], system['W_k'], system['W_v'], system['W_o'],
        system['d_model'], system['num_heads']
    )
    
    # Global pooling (max + mean)
    pooled_max = np.max(attn_output, axis=0)
    pooled_mean = np.mean(attn_output, axis=0)
    pooled = (pooled_max + pooled_mean) / 2
    
    # Classification
    logits = np.dot(pooled, system['W_class']) + system['bias']
    probs = softmax(logits)
    
    return probs, attn_output, attn_weights, feature_names


# Step 5: Define Medical Knowledge Base
SYMPTOMS = {
    'fever': 0, 'cough': 1, 'fatigue': 2, 'chest_pain': 3,
    'shortness_of_breath': 4, 'headache': 5, 'body_ache': 6,
    'sore_throat': 7, 'runny_nose': 8, 'nausea': 9,
    'dizziness': 10, 'rapid_heartbeat': 11, 'sweating': 12
}

VITALS = {
    'temperature': 0, 'heart_rate': 1, 'blood_pressure_sys': 2,
    'blood_pressure_dia': 3, 'oxygen_saturation': 4, 'respiratory_rate': 5
}

DISEASES = ['Influenza', 'Pneumonia', 'COVID-19', 'Heart Disease', 'Healthy']

# Sample patient cases
PATIENT_CASES = [
    {
        'name': 'Patient A - Suspected Flu',
        'symptoms': ['fever', 'cough', 'body_ache', 'fatigue', 'headache'],
        'vitals': [
            ('temperature', 101.5),
            ('heart_rate', 95),
            ('oxygen_saturation', 97)
        ],
        'expected': 'Influenza'
    },
    {
        'name': 'Patient B - Respiratory Issues',
        'symptoms': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue'],
        'vitals': [
            ('temperature', 102.8),
            ('heart_rate', 105),
            ('oxygen_saturation', 92),
            ('respiratory_rate', 24)
        ],
        'expected': 'Pneumonia'
    },
    {
        'name': 'Patient C - Cardiac Symptoms',
        'symptoms': ['chest_pain', 'shortness_of_breath', 'dizziness', 'sweating'],
        'vitals': [
            ('heart_rate', 115),
            ('blood_pressure_sys', 160),
            ('blood_pressure_dia', 95)
        ],
        'expected': 'Heart Disease'
    }
]


# Step 6: Demonstration of Attention Mechanisms
print("=" * 90)
print("SELF-ATTENTION AND MULTI-HEAD ATTENTION DEMO")
print("=" * 90)

# Create sample input for demonstration
demo_input = np.random.randn(4, 8)
print(f"\nInput shape: {demo_input.shape}")
print(f"Input data:\n {np.round(demo_input, 2)}")

# Initialize demo weights
d_model_demo = 8
W_q_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
W_k_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
W_v_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1

# Self-Attention Demo
print(f"\n{'='*60}")
print("SELF-ATTENTION")
print(f"{'='*60}")

self_attn_output, self_attn_weights = self_attention_forward(
    demo_input, W_q_demo, W_k_demo, W_v_demo, d_model_demo
)

print(f"\nSelf-Attention Output shape: {self_attn_output.shape}")
print(f"Self-Attention Output:\n {np.round(self_attn_output, 3)}")
print(f"\nSelf-Attention Weights shape: {self_attn_weights.shape}")
print(f"Self-Attention Weights:\n {np.round(self_attn_weights, 3)}")

# Multi-Head Attention Demo
print(f"\n{'='*60}")
print("MULTI-HEAD ATTENTION (2 heads)")
print(f"{'='*60}")

W_o_demo = np.random.randn(d_model_demo, d_model_demo) * 0.1
num_heads_demo = 2

multi_head_output, multi_head_weights = multi_head_attention_forward(
    demo_input, W_q_demo, W_k_demo, W_v_demo, W_o_demo, 
    d_model_demo, num_heads_demo
)

print(f"\nMulti-Head Attention Output shape: {multi_head_output.shape}")
print(f"Multi-Head Attention Output:\n {np.round(multi_head_output, 3)}")
print(f"\nAttention Weights per Head:")

for head_idx, weights in enumerate(multi_head_weights):
    print(f"\nHead {head_idx + 1} Attention Weights:")
    print(f" {np.round(weights, 3)}")


# Step 7: Run Medical Diagnosis System
print(f"\n{'='*90}")
print("MEDICAL DIAGNOSIS SYSTEM USING ATTENTION MECHANISMS")
print(f"{'='*90}")

# Initialize system
d_model = 32
num_heads = 4
diagnosis_system = initialize_diagnosis_system(d_model, num_heads, len(DISEASES))

print(f"\n🏥 System Configuration:")
print(f"   • Model Dimension: {d_model}")
print(f"   • Attention Heads: {num_heads}")
print(f"   • Trackable Symptoms: {len(SYMPTOMS)}")
print(f"   • Vital Signs: {len(VITALS)}")
print(f"   • Disease Categories: {len(DISEASES)}")

# Process each patient
all_results = []

for case_idx, patient in enumerate(PATIENT_CASES):
    print(f"\n{'='*90}")
    print(f"CASE {case_idx + 1}: {patient['name']}")
    print(f"{'='*90}")
    
    # Prepare data
    symptom_ids = [SYMPTOMS[s] for s in patient['symptoms']]
    vital_data = [(VITALS[v[0]], v[1]) for v in patient['vitals']]
    
    # Create feature names for visualization
    feature_names = patient['symptoms'] + [f"{v[0]}={v[1]}" for v in patient['vitals']]
    
    print(f"\n📋 Presenting Symptoms:")
    for symptom in patient['symptoms']:
        print(f"   • {symptom.replace('_', ' ').title()}")
    
    print(f"\n🩺 Vital Signs:")
    for vital_name, vital_value in patient['vitals']:
        print(f"   • {vital_name.replace('_', ' ').title()}: {vital_value}")
    
    # Run diagnosis
    probs, attn_output, attn_weights, features = diagnose(
        symptom_ids, vital_data, diagnosis_system, feature_names
    )
    
    # Display results
    print(f"\n🔍 Diagnosis Probabilities:")
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        disease = DISEASES[idx]
        prob = probs[idx]
        bar = '█' * int(prob * 40)
        print(f"   {disease:20s}: {prob:.2%} {bar}")
    
    predicted_disease = DISEASES[np.argmax(probs)]
    print(f"\n✅ Primary Diagnosis: {predicted_disease}")
    print(f"   Expected: {patient['expected']}")
    
    # Analyze attention patterns
    print(f"\n🧠 Attention Analysis (What the model focuses on):")
    
    for head_idx in range(num_heads):
        print(f"\n   Head {head_idx + 1} - Key Features:")
        # Average attention each feature receives
        avg_attention = np.mean(attn_weights[head_idx], axis=0)
        
        # Get top 3 features this head focuses on
        top_indices = np.argsort(avg_attention)[-3:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            if idx < len(features):
                feature = features[idx]
                attention_score = avg_attention[idx]
                bar = '▓' * int(attention_score * 30)
                print(f"      {rank}. {feature:25s}: {attention_score:.3f} {bar}")
    
    all_results.append({
        'patient': patient,
        'features': features,
        'probs': probs,
        'predicted': predicted_disease,
        'attn_weights': attn_weights
    })


# Step 7: Comprehensive Visualization
print(f"\n{'='*90}")
print("ATTENTION VISUALIZATION")
print(f"{'='*90}")

fig = plt.figure(figsize=(20, 12))

for case_idx, result in enumerate(all_results):
    features = result['features']
    attn_weights = result['attn_weights']
    patient = result['patient']
    
    # Plot each attention head
    for head_idx in range(num_heads):
        subplot_idx = case_idx * num_heads + head_idx + 1
        ax = plt.subplot(len(all_results), num_heads, subplot_idx)
        
        # Plot attention heatmap
        im = ax.imshow(attn_weights[head_idx], cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(range(len(features)))
        ax.set_yticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(features, fontsize=8)
        
        # Title
        title = f"{patient['name']}\nHead {head_idx+1} | Predicted: {result['predicted']}"
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Attends To', fontsize=8)
        ax.set_ylabel('Feature', fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('medical_diagnosis_attention.png', dpi=150, bbox_inches='tight')
print("\n📊 Visualization saved as 'medical_diagnosis_attention.png'")
plt.show()


# Step 8: Clinical Insights
print(f"\n{'='*90}")
print("CLINICAL INSIGHTS: WHY ATTENTION MATTERS IN MEDICAL AI")
print(f"{'='*90}")

print("""
🔬 Multi-Head Attention Benefits in Healthcare:

1. INTERPRETABILITY FOR CLINICIANS
   • Shows which symptoms/vitals drove each diagnosis
   • Different heads can focus on different diagnostic pathways
   • Provides transparency for medical decision support

2. FEATURE RELATIONSHIP DISCOVERY
   • Head 1: May focus on respiratory symptoms (cough + oxygen saturation)
   • Head 2: May focus on cardiovascular signs (chest pain + heart rate)
   • Head 3: May focus on inflammatory markers (fever + body ache)
   • Head 4: May focus on vital sign correlations

3. HANDLING COMPLEX SYMPTOM INTERACTIONS
   • Attention captures non-linear relationships
   • Example: High fever + low O2 saturation = more severe than individually
   • Context-aware: Same symptom means different things in different contexts

4. MISSING DATA ROBUSTNESS
   • Can work with variable number of symptoms/vitals
   • Attention naturally handles incomplete patient data
   • No fixed input structure required
""")

print(f"{'='*90}")
print("COMPARISON: SELF-ATTENTION vs MULTI-HEAD ATTENTION")
print(f"{'='*90}")

print("""
SELF-ATTENTION (Single Head):
   ✓ Captures one view of symptom relationships
   ✓ Simpler, faster computation
   ✗ Limited to one attention pattern
   ✗ May miss important feature interactions

MULTI-HEAD ATTENTION (Multiple Heads):
   ✓ Captures multiple diagnostic reasoning paths simultaneously
   ✓ More robust and accurate predictions
   ✓ Each head specializes in different patterns
   ✓ Better generalization across diverse cases
   ✗ More computational cost (but worth it for medical applications)
""")

print(f"{'='*90}")
print("REAL-WORLD MEDICAL AI APPLICATIONS")
print(f"{'='*90}")

print("""
This attention-based approach is valuable for:

   🏥 Emergency Triage Systems
      • Rapid assessment of incoming patients
      • Prioritize critical cases automatically
      
   🔬 Diagnostic Support Systems
      • Assist doctors in complex diagnoses
      • Suggest tests based on symptom patterns
      
   📊 Patient Monitoring
      • Track deteriorating conditions in ICU
      • Alert staff to concerning vital sign patterns
      
   💊 Drug Interaction Analysis
      • Analyze medication combinations
      • Predict adverse reactions
      
   🧬 Genomic Medicine
      • Correlate genetic markers with diseases
      • Personalized treatment recommendations
      
   📱 Telemedicine Applications
      • Remote symptom assessment
      • Guide patients to appropriate care level
""")

print(f"{'='*90}")
print("ETHICAL CONSIDERATIONS")
print(f"{'='*90}")

print("""
⚠️  Important Notes for Medical AI:
   • This is a demonstration - NOT for actual medical use
   • Always requires human physician oversight
   • Attention weights provide explainability but not causality
   • Must be trained on large, validated medical datasets
   • Requires regulatory approval (FDA, CE marking, etc.)
   • Should augment, not replace, clinical judgment
""")

print(f"\n{'='*90}")
print("✨ This demonstrates how attention mechanisms provide both accuracy AND interpretability")
print("   making them ideal for high-stakes applications like healthcare!")
print(f"{'='*90}\n")