import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Intelligent Player Churn Prediction System", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2c3e50; text-align: center;}
    .stButton>button {color: white; background-color: #e74c3c;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        return joblib.load('models/logistic_regression.pkl'), joblib.load('models/decision_tree.pkl'), joblib.load('models/scalers.pkl'), joblib.load('models/label_encoders.pkl')
    except Exception as e:
        st.error(f"Model files missing: {e}")
        return None, None, None, None

log_model, dt_model, scaler, les = load_assets()

def process(df, les, scaler):
    df_p = df.copy()
    for col in df_p.select_dtypes(include=np.number).columns: df_p[col].fillna(df_p[col].mean(), inplace=True)
    for col in df_p.select_dtypes(include='object').columns: df_p[col].fillna(df_p[col].mode()[0], inplace=True)
    
    req = ['Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours', 'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
    if set(req) - set(df_p.columns): return None
    df_p = df_p[req]

    for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
        if col in df_p:
            le = les.get(col)
            if le:
                df_p.loc[~df_p[col].isin(le.classes_), col] = le.classes_[0]
                df_p[col] = le.transform(df_p[col])
            else:
                df_p[col] = pd.factorize(df_p[col])[0]

    return scaler.transform(df_p), df_p

st.title("Player Churn Prediction System")
st.markdown("### Process Overview")
st.write("1. **Data Upload**: User provides a CSV file.")
st.write("2. **Preprocessing**: Missing values are imputed, and categorical features are encoded.")
st.write("3. **Prediction**: The selected model predicts churn probability.")
st.write("4. **Risk Assessment**: Players are categorized as Low (<30%), Medium (30-70%), or High (>70%) risk.")


with st.sidebar:
    st.header("Upload Data")
    f = st.file_uploader("Upload CSV", type=["csv"])
    m = st.selectbox("Model", ["Logistic Regression", "Decision Tree"])

if f:
    try:
        df = pd.read_csv(f)
        if log_model and dt_model:
            st.markdown("---")
            st.subheader("1. Data Processing")
            X_s, X_us = process(df, les, scaler)
            st.success("Data processed successfully! Imputed missing values and encoded categorical features.")
            
            if X_s is not None:
                st.subheader("2. Prediction & Risk Assessment")
                mod = log_model if m == "Logistic Regression" else dt_model
                inp = X_s if m == "Logistic Regression" else X_us
                
                pred = mod.predict(inp)
                prob = mod.predict_proba(inp)[:, 1]
                
                res = df[['PlayerID']].copy() if 'PlayerID' in df else df.reset_index()[['index']].rename(columns={'index': 'PlayerID'})
                res['Churn'], res['Prob'] = pred, prob
                res['Risk'] = ['Low' if p < 0.3 else 'Medium' if p <= 0.7 else 'High' for p in prob]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(res))
                c2.metric("High Risk", len(res[res['Risk'] == 'High']))
                c3.metric("Avg Prob", f"{prob.mean():.2%}")
                
                def highlight_risk(val):
                    bg_color = '#d4edda' if val == 'Low' else '#fff3cd' if val == 'Medium' else '#f8d7da'
                    text_color = '#155724' if val == 'Low' else '#856404' if val == 'Medium' else '#721c24'
                    return f'background-color: {bg_color}; color: {text_color}; font-weight: bold'
                
                styled_df = res.style.format({'Prob': '{:.2%}'}).applymap(highlight_risk, subset=['Risk'])
                st.dataframe(styled_df)
                fig, ax = plt.subplots()
                v = res['Risk'].value_counts()
                sns.barplot(x=v.index, y=v.values, palette={"Low": "green", "Medium": "orange", "High": "red"}, ax=ax)
                st.pyplot(fig)
                
                if m == "Decision Tree":
                    imp = pd.DataFrame({'Feat': X_us.columns, 'Imp': mod.feature_importances_}).sort_values('Imp', ascending=False).head(5)
                    fig2, ax2 = plt.subplots()
                    sns.barplot(x='Imp', y='Feat', data=imp, ax=ax2)
                    st.pyplot(fig2)
                
                st.download_button("Download", res.to_csv(index=False).encode('utf-8'), "pred.csv", "text/csv")
    except Exception as e: st.error(f"Error: {e}")
else: st.info("Upload CSV.")
