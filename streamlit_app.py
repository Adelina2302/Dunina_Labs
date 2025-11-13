import streamlit as st

pg = st.navigation(
    {
        "Labs": [
            st.Page("Labs/Lab9.py", title="Lab9"),
            st.Page("Labs/Lab8.py", title="Lab8"),
            st.Page("Labs/Lab6.py", title="Lab6"),
            st.Page("Labs/Lab5.py", title="Lab5"),
            st.Page("Labs/Lab4.py", title="Lab4"),
            st.Page("Labs/Lab3.py", title="Lab3"), 
            st.Page("Labs/Lab2.py", title="Lab2"),
            st.Page("Labs/Lab1.py", title="Lab1"),
        ]
    }
)

pg.run()
