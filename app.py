import streamlit as st
import json
import yaml
from utils.file_handler import extract_text
from utils.evaluator import calculate_accuracy, save_results
from models.openai_model import OpenAIModel
from models.anthropic_model import AnthropicModel

# ---------------------------
# STREAMLIT APP STARTS HERE
# ---------------------------
st.set_page_config(page_title="LLM Evaluation Tool", layout="wide")
st.title("📊 LLM Entity Extraction Evaluation Tool")

st.markdown("""
This tool evaluates and compares multiple Large Language Models (LLMs)  
for **entity extraction** from uploaded documents.
""")

uploaded_file = st.file_uploader("📁 Upload a Document", type=["pdf", "docx", "txt"])
config_file = st.file_uploader("⚙️ Upload Config File (JSON/YAML)", type=["json", "yaml"])

selected_models = st.multiselect(
    "🧠 Select Models to Evaluate",
    ["OpenAI GPT-4o", "Claude 3 Sonnet"],
    default=["OpenAI GPT-4o"]
)

run_button = st.button("🚀 Run Evaluation")

if run_button:
    if not uploaded_file or not config_file:
        st.error("⚠️ Please upload both a document and a configuration file before running.")
    else:
        with st.spinner("Processing... please wait ⏳"):
            # Step 1: Extract text
            text = extract_text(uploaded_file)

            # Step 2: Load configuration
            if config_file.name.endswith(".json"):
                config = json.load(config_file)
            else:
                config = yaml.safe_load(config_file)

            entities = config["entities"]
            ground_truth = config["ground_truth"]

            results = []

            # Step 3: Run selected models
            if "OpenAI GPT-4o" in selected_models:
                model = OpenAIModel()
                output, latency, tokens, cost = model.extract_entities(text, entities)
                acc = calculate_accuracy(output, ground_truth)
                results.append({
                    "Model": "GPT-4o",
                    "Accuracy (%)": acc,
                    "Latency (s)": round(latency, 2),
                    "Tokens Used": tokens,
                    "Cost ($)": round(cost, 6)
                })

            if "Claude 3 Sonnet" in selected_models:
                model = AnthropicModel()
                output, latency, tokens, cost = model.extract_entities(text, entities)
                acc = calculate_accuracy(output, ground_truth)
                results.append({
                    "Model": "Claude 3 Sonnet",
                    "Accuracy (%)": acc,
                    "Latency (s)": round(latency, 2),
                    "Tokens Used": tokens,
                    "Cost ($)": round(cost, 6)
                })

            # Step 4: Save & display results
            df = save_results(results)
            st.success("✅ Evaluation Complete!")
            st.subheader("📈 Model Comparison Results")
            st.dataframe(df)
            st.bar_chart(df.set_index("Model")[["Accuracy (%)", "Latency (s)"]])
            st.download_button(
                label="⬇️ Download CSV Report",
                data=df.to_csv(index=False),
                file_name="llm_evaluation_report.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Developed by Radha Mahendra Gharge • Guide: Vaishali Kandekar")
