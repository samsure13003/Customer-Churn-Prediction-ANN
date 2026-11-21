import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Main Title */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #A0AEC0;
        margin-bottom: 30px;
    }

    /* Section Headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 15px;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #1E293B, #334155);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #475569;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
        color: #60A5FA;
    }

    .prediction-label {
        font-size: 16px;
        color: #CBD5E1;
    }

    /* Input Labels */
    label {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
    }

    /* Divider */
    hr {
        border-color: #334155;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND ARTIFACTS
# ==========================================

artifacts_dir = Path(__file__).resolve().parent / 'artifacts'


@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        artifacts_dir / 'model.h5'
    )

    with open(artifacts_dir / 'label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open(artifacts_dir / 'onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open(artifacts_dir / 'scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer retention and churn risk analysis dashboard.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# MAIN LAYOUT
# ==========================================

left_column, right_column = st.columns([1.3, 1])


# ==========================================
# CUSTOMER INPUT SECTION
# ==========================================

with left_column:

    st.markdown(
        '<div class="section-header">👤 Customer Information</div>',
        unsafe_allow_html=True
    )

    # Personal Information
    st.markdown("### Personal Details")

    col1, col2 = st.columns(2)

    with col1:

        geography = st.selectbox(
            "Geography",
            onehot_encoder_geo.categories_[0]
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            label_encoder_gender.classes_
        )

    age = st.slider(
        "Age",
        18,
        92,
        35
    )

    st.divider()

    # Financial Information
    st.markdown("### Financial Details")

    col3, col4 = st.columns(2)

    with col3:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col4:

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        tenure = st.slider(
            "Tenure (Years)",
            0,
            10,
            5
        )

    st.divider()

    # Account Information
    st.markdown("### Account Details")

    col5, col6 = st.columns(2)

    with col5:

        num_of_products = st.slider(
            "Number of Products",
            1,
            4,
            1
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col6:

        is_active_member = st.selectbox(
            "Active Member?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )


# ==========================================
# PREDICTION SECTION
# ==========================================

with right_column:

    st.markdown(
        '<div class="section-header">📈 Prediction Results</div>',
        unsafe_allow_html=True
    )

    # Prepare Input Data
    input_data = pd.DataFrame({

        'CreditScore': [credit_score],

        'Gender': [
            label_encoder_gender.transform([gender])[0]
        ],

        'Age': [age],

        'Tenure': [tenure],

        'Balance': [balance],

        'NumOfProducts': [num_of_products],

        'HasCrCard': [has_cr_card],

        'IsActiveMember': [is_active_member],

        'EstimatedSalary': [estimated_salary]

    })

    # One Hot Encoding
    geo_encoded = onehot_encoder_geo.transform(
        [[geography]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(
            ['Geography']
        )
    )

    # Combine Encoded Features
    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geo_encoded_df
        ],
        axis=1
    )

    # Scale Input Data
    input_data_scaled = scaler.transform(input_data)

    # ANN Prediction
    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_proba = float(prediction[0][0])

    probability_percentage = prediction_proba * 100


    # ======================================
    # DISPLAY CHURN PROBABILITY
    # ======================================

    st.markdown(
    f"""
    <div class="prediction-card">
        <div class="prediction-label">
            CUSTOMER CHURN PROBABILITY
        </div>
        <div class="prediction-value">
            {probability_percentage:.2f}%
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


    # ======================================
    # PROBABILITY PROGRESS BAR
    # ======================================

    st.markdown("### Risk Probability")

    st.progress(prediction_proba)


    # ======================================
    # PREDICTION STATUS
    # ======================================

    st.markdown("### Prediction Status")

    if prediction_proba > 0.5:

        st.error(
            "⚠️ High Risk: Customer is likely to churn."
        )

        st.markdown("### Recommended Actions")

        st.markdown("""
        - 📞 Contact the customer.
        - 🎁 Offer personalized discounts.
        - 🤝 Provide better customer support.
        - 🔍 Investigate customer dissatisfaction.
        """)

    else:

        st.success(
            "✅ Low Risk: Customer is unlikely to churn."
        )

        st.markdown("### Recommended Actions")

        st.markdown("""
        - ⭐ Maintain customer engagement.
        - 📈 Continue providing quality services.
        - 😊 Monitor customer satisfaction.
        """)


    # ======================================
    # MODEL INFORMATION
    # ======================================

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.write("**Algorithm:** Artificial Neural Network (ANN)")

    st.write("**Framework:** TensorFlow / Keras")

    st.write("**Preprocessing:** StandardScaler + Encoding")

    st.write("**Classification Threshold:** 0.50")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748B;
        padding:15px;
        font-size:14px;
    ">

        Built with ❤️ using Streamlit, TensorFlow and Scikit-learn


        Customer Churn Prediction | Machine Learning Project

    </div>
    """,
    unsafe_allow_html=True
)
