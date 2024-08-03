import streamlit as st

st.set_page_config(
    page_title="Log Out",
    page_icon="👋",
)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.write("You are logged out")
    st.stop()

st.session_state.logged_in = False

st.success("You have been logged out successfully!")

st.rerun()
