import streamlit as st
from app.ml_app import run_ml_app

def main():
    st.sidebar.title("✈️ Flight Delay App")
    menu = ["Home", "Prediction"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Home":
        st.title("Flight Delay Prediction App")
        st.write("Selamat datang! Gunakan menu *Prediction* di sidebar untuk memprediksi keterlambatan penerbangan.")
    elif choice == "Prediction":
        run_ml_app()

if __name__ == "__main__":
    main()
