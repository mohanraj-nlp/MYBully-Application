import streamlit as st
import pandas as pd
from inference import predict_emotion, predict_hate, predict_sentiment, predict_cyberbully, get_risk_level
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")
st.title("🚨 Cyberbullying Detector for Tweets")

tabs = st.tabs(["🧵 Single Tweet", "📁 Bulk Upload"])

# ================================
# SINGLE TWEET TAB
# ================================
with tabs[0]:
    st.subheader("🔎 Analyze a Single Tweet")
    tweet = st.text_area("✏️ Enter your tweet below:", height=150, placeholder="Type a tweet here...")

    if st.button("Analyze Tweet"):
        if not tweet.strip():
            st.warning("Please enter a tweet.")
        else:
            with st.spinner("Predicting..."):
                emotion = predict_emotion(tweet)
                hate = predict_hate(tweet)
                sentiment = predict_sentiment(tweet)
                cyberbully = predict_cyberbully(tweet, emotion, hate, sentiment)
                risk = get_risk_level(emotion, hate, sentiment)

            st.success("Prediction Complete ✅")

            with st.expander("📊 View Results"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Emotion", emotion)
                col2.metric("Hate Speech", hate)
                col3.metric("Cyberbully", cyberbully)
                col4.metric("Risk Level", risk)

            st.markdown(f"""
            <strong style="font-size:20px;">🔍 Result:</strong><br>
            <div style="padding:1rem; border-left:5px solid {'red' if cyberbully=='Cyberbully' else 'green'}; font-size:17px; line-height:1.8;">
            <strong style="font-size:14px;">Emotion:</strong> <code>{emotion}</code><br>
            <strong style="font-size:14px;">Hate Speech:</strong> <code>{hate}</code><br>
            <strong style="font-size:14px;">Tweet Classified as:</strong> <code>{cyberbully}</code><br>
            <strong style="font-size:14px;">Risk Level:</strong> <code>{risk}</code><br>
            </div>
            """, unsafe_allow_html=True)
            
# ================================
# BULK UPLOAD TAB
# ================================
with tabs[1]:
    st.subheader("📤 Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'tweet' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "tweet" not in df.columns:
            st.error("CSV must have a 'tweet' column.")
        else:
            st.success(f"{len(df)} tweets loaded. Processing...")

            results = []
            with st.spinner("Predicting in batch..."):
                for tweet in df["tweet"]:
                    emo = predict_emotion(tweet)
                    hate = predict_hate(tweet)
                    sentiment = predict_sentiment(tweet)
                    cyber = predict_cyberbully(tweet, emo, hate, sentiment)
                    risk = get_risk_level(emo, hate, sentiment)
                    results.append({
                        "Tweet": tweet,
                        "Emotion": emo,
                        "Hate": hate,
                        "Cyberbully": cyber,
                        "Risk": risk
                    })

            results_df = pd.DataFrame(results)
            st.markdown("### 📋 Prediction Table")
            st.dataframe(results_df, use_container_width=True)

            # --- Filters ---
            st.markdown("### 🧪 Filter Results")
            cyber_filter = st.selectbox("Filter by Cyberbully", ["All"] + sorted(results_df["Cyberbully"].unique()))
            emo_filter = st.selectbox("Filter by Emotion", ["All"] + sorted(results_df["Emotion"].unique()))
            hate_filter = st.selectbox("Filter by Hate Speech", ["All"] + sorted(results_df["Hate"].unique()))
            risk_filter = st.selectbox("Filter by Risk", ["All"] + sorted(results_df["Risk"].unique()))

            filtered_df = results_df.copy()
            if cyber_filter != "All":
                filtered_df = filtered_df[filtered_df["Cyberbully"] == cyber_filter]
            if emo_filter != "All":
                filtered_df = filtered_df[filtered_df["Emotion"] == emo_filter]
            if hate_filter != "All":
                filtered_df = filtered_df[filtered_df["Hate"] == hate_filter]
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df["Risk"] == risk_filter]

            st.dataframe(filtered_df, use_container_width=True)

            # --- Download ---
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Filtered Results CSV", csv, "cyberbully_results.csv", "text/csv")

            # --- Charts ---
            st.markdown("### 📈 Distribution Charts")
            col1, col2 = st.columns(2)

            with col1:
                emo_counts = results_df["Emotion"].value_counts().reset_index()
                emo_counts.columns = ["Emotion", "Count"]
                fig = px.pie(emo_counts, names="Emotion", values="Count", title="Emotion Distribution")
                st.plotly_chart(fig)

            with col2:
                cyber_counts = results_df["Cyberbully"].value_counts().reset_index()
                cyber_counts.columns = ["Label", "Count"]
                fig2 = px.bar(cyber_counts, x="Label", y="Count", title="Cyberbullying Classification Count")
                st.plotly_chart(fig2)

            st.markdown("### ☁️ Word Cloud of Cyberbullying Tweets")
            cyber_text = " ".join(results_df[results_df["Cyberbully"] == "Cyberbully"]["Tweet"])
            if cyber_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cyber_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No Cyberbully tweets found to generate word cloud.")
