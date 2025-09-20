# import packages
import streamlit as st
import pandas as pd
import re
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.markdown("""
<style>
/* Focus / active (after click, when dropdown open or focused) */
.stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: #1E90FF !important;
    box-shadow: 0 0 0 1px #1E90FF;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def model_response(userPrompt, generation_config):
    if userPrompt not in (None, ""):
        response = model.generate_content(
            userPrompt,
            generation_config=generation_config
        )
        st.write(response.text.strip())

def clean_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "..", "..", "data", "customer_reviews.csv")
    return csv_path


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Data"):
        try:
            csv_path = get_dataset_path()
            st.session_state['df'] = pd.read_csv(csv_path)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")

with col2:
    # Temperature slider
    temperature = st.slider(
        "Select temperature (creativity level):",
        0.0, 1.0, 0.7, 0.01,
        help="Higher = more creative. Lower = more focused."
    )
    # Build generation config AFTER temperature is set
    generation_config = {
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 150
    }

    if 'df' in st.session_state and st.button("Run Sentiment Analysis"):
        df = st.session_state['df'].copy()
        if 'PRODUCT' not in df.columns:
            st.error("PRODUCT column not found.")
        else:
            with st.spinner("Analyzing sentiment per product..."):
                # Prepare product -> sample text map
                text_col_candidates = [c for c in df.columns if c.lower() in ("review", "text", "comment", "feedback")]
                review_col = text_col_candidates[0] if text_col_candidates else None
                product_samples = {}
                for prod, sub in df.groupby('PRODUCT'):
                    if review_col:
                        samples = sub[review_col].astype(str).head(8).tolist()
                    else:
                        # fallback: concatenate first row stringified data
                        samples = [(" ".join(map(str, sub.head(1).values.flatten())))]
                    joined = " | ".join(clean_text(s) for s in samples if isinstance(s, str))
                    product_samples[prod] = joined[:2000]  # guard length

                prompt_lines = ["Classify overall sentiment for each product as exactly 'Positive' or 'Negative'.",
                                "Return ONLY a strict JSON object: {\"PRODUCT_NAME\": \"Positive\"|\"Negative\", ...}.",
                                "Do not include explanations.",
                                "Data:"]
                for p, txt in product_samples.items():
                    prompt_lines.append(f"{p}: {txt}")
                prompt = "\n".join(prompt_lines)

                try:
                    resp = model.generate_content(prompt, generation_config=generation_config)
                    raw = resp.text.strip()
                    # Extract JSON
                    json_str_match = re.search(r'\{.*\}', raw, re.DOTALL)
                    if not json_str_match:
                        raise ValueError("No JSON object found in model response.")
                    sentiments_map = json.loads(json_str_match.group(0))
                    # Map back
                    df['PRODUCT_SENTIMENT'] = df['PRODUCT'].map(lambda p: sentiments_map.get(str(p), "Unknown"))
                except Exception as e:
                    st.warning(f"Model classification failed ({e}). Falling back to numeric SENTIMENT_SCORE if available.")
                    if 'SENTIMENT_SCORE' in df.columns:
                        df['PRODUCT_SENTIMENT'] = df['SENTIMENT_SCORE'].apply(lambda v: "Positive" if v >= 0 else "Negative")
                    else:
                        df['PRODUCT_SENTIMENT'] = "Unknown"

                st.session_state['df'] = df
                st.success("Sentiment analysis complete.")

if 'df' in st.session_state:
    st.subheader("Filter by product category")
    categories = st.session_state['df']['PRODUCT'].unique()
    selected_category = st.selectbox("Select a category", ['ALL_PRODUCTS'] + list(categories))
    if selected_category == 'ALL_PRODUCTS':
        filtered_df = st.session_state['df']
    else:
        filtered_df = st.session_state['df'][st.session_state['df']['PRODUCT'] == selected_category]
    st.dataframe(filtered_df)

