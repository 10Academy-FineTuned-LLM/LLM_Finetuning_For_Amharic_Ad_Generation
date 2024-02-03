import streamlit as st
import pandas as pd 


# Initialize the Streamlit app
st.set_page_config(page_title="Ad App", page_icon=":guardsman:", layout="wide")



# Create a container for the input box and button

container = st.container()


"# welcome to Group two's LLM fine tuned model \n To generate an AD please upload your document below. "

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")


if uploaded_pdf is not None:

    with st.container():

        st.write("PDF Information:")

        st.markdown(f"**File Name:** {uploaded_pdf.name}")

        st.markdown(f"**File Type:** {uploaded_pdf.type}")

        st.markdown(f"**File Size:** {uploaded_pdf.size} bytes")



# Create the sidebar
st.sidebar.title("Ad Options")
add_column1 = st.sidebar.button("Ad generator")
add_column2 = st.sidebar.button("Ad checker")
add_column3 = st.sidebar.button("Ad type checker")



# Check if the button was clicked
if add_column1:
    # Create a container for the input box and button

    container = st.container()


    # Create an input box for the user query

    user_query = container.text_input("Enter your prompt here to generate an advetisement:")


    # Create a button to submit the query

    submit_query = container.button("Generate")


    # Check if the submit button was clicked

    if submit_query:

        # Call the backend function to check the ad type

        ad_type = check_ad_type(user_query)

        

        # Display the result

        container.write(f"Ad type: {ad_type}")
    

# Check if the button was clicked
if add_column2:
    # Create a container for the input box and button

    container = st.container()


    # Create an input box for the user query

    user_query = container.text_input("Enter your string here to check if it is an advetisement or not:")


    # Create a button to submit the query

    submit_query = container.button("Check")


    # Check if the submit button was clicked

    if submit_query:

        # Call the backend function to check the ad type

        ad_type = check_ad_type(user_query)

        

        # Display the result

        container.write(f"Ad type: {ad_type}")
    

# Check if the button was clicked
if add_column3:
    # Create a container for the input box and button

    container = st.container()


    # Create an input box for the user query

    user_query = container.text_input("Enter your advertisement here to check its type:")


    # Create a button to submit the query

    submit_query = container.button("Check")


    # Check if the submit button was clicked

    if submit_query:

        # Call the backend function to check the ad type

        ad_type = check_ad_type(user_query)

        

        # Display the result

        container.write(f"Ad type: {ad_type}")
