
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝"
)

st.title("📝 AI Text Summarizer")
st.write("Turn long text into short, easy-to-read summaries.")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

text = st.text_area(
    "Paste your text here",
    height=250,
    placeholder="Enter an article or long paragraph..."
)

if st.button("✨ Summarize"):
    if len(text.split()) < 40:
        st.warning("Please enter at least 40 words.")
    else:
        try:
            with st.spinner("AI is summarizing your text..."):
                summarizer = load_model()
                result = summarizer(
                    text[:5000],
                    max_length=100,
                    min_length=25,
                    do_sample=False
                )

            st.subheader("Your Summary")
            st.write(result[0]["summary_text"])

        except Exception as error:
            st.error("Something went wrong. Please try again.")
            st.caption(str(error))

st.divider()
st.caption("Built with Python, Streamlit and Hugging Face.")
