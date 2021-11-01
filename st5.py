import st7
import st6
import streamlit as st
PAGES = {
    "Classification Explanatory Tool": st7,
    "Prediction App": st6
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()