import streamlit as st
from streamlit_local_storage import LocalStorage

st.write("Testing local storage")
localS = LocalStorage()

if st.button("Set item"):
    localS.setItem("testKey", "testValue")
    
if st.button("Get item"):
    val = localS.getItem("testKey")
    st.write(val)
