import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier

# Загрузка модели и продуктов
@st.cache_resource
def load_model():
    return CatBoostClassifier().load_model("model/catboost_model_ROC8.cbm")

@st.cache_data
def load_products():
    return pd.read_csv("data/products.csv")

model = load_model()
products = load_products()

st.title("📊 Финансовая рекомендательная система")

st.header("Введите данные пользователя")

age = st.slider("Возраст", 18, 80, 30)
income = st.number_input("Доход", min_value=10000, max_value=500000, value=80000, step=5000)
has_mortgage = st.selectbox("Есть ипотека?", ["Нет", "Да"])
goal = st.selectbox("Финансовая цель", ["накопление", "пенсия", "покупка жилья"])

# Обработка
from catboost import Pool

if st.button("Получить рекомендации"):
    user_data = pd.DataFrame([{
        "age": age,
        "income": income,
        "has_mortgage": 1 if has_mortgage == "Да" else 0,
        "goal_x": goal
    }])

    df = user_data.assign(key=1).merge(products.assign(key=1), on="key").drop("key", axis=1)
    df.rename(columns={"goal": "goal_y"}, inplace=True)

    # Те же признаки
    X = df[["age", "income", "has_mortgage", "min_income", "risk_score", "goal_x", "goal_y", "type"]]

    # Передаем категориальные признаки вручную через Pool
    cat_features = ["goal_x", "goal_y", "type"]
    pool = Pool(data=X, cat_features=cat_features)

    df["score"] = model.predict_proba(pool)[:, 1]

    recommendations = df[["type", "goal_y", "risk_score", "score"]].sort_values("score", ascending=False).head(5)

    st.subheader("🔮 Рекомендованные продукты:")
    st.dataframe(recommendations.reset_index(drop=True))

