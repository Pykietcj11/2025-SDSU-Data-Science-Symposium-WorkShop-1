
import streamlit as st
# Optional
st.set_page_config(layout="wide")

st.title("Hello, World! This is Cameron!")

st.header('This is a demonstration of how to build an app with streamlit.')

st.caption('We are currently going over the different kind of text elements that can be used.')

st.markdown('# Such as a markdown text')

st.code('''
        #Or even lines of code. 
        x = 7
        y = 10
        z = x+y
        print(z)
        ''')
# and Add a divider
st.divider()

col1, col2 = st.columns(2, gap = 'large')
col1.subheader("Left side?")
col2.subheader("Right side?")

chat = col1.chat_input('Have you ever been to Mount Rushmoore?')

if chat == 'yes':
    st.write("So have I! Though it was underwhelming in my opinion.")
