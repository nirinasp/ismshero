import streamlit as st

st.set_page_config(
    page_title="Login",
    page_icon="👋"
)

def login(username, password):
    """Validate the login credentials."""
    if username == st.secrets["ADMIN_USERNAME"] and password == st.secrets["ADMIN_PASSWORD"]:
        return True
    return False

def form():
    st.title("Login")

    # Check if the user is already logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # If logged in, display the main content
    if st.session_state.logged_in:
        st.write("You are logged in!")
    else:
        # Display the login form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

form()
