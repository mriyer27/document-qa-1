import streamlit as st

# ---------------------------------------------------------------------------
# Part A: multi-page 'Lab' application
# Each lab lives in its own file; this file just wires up the navigation.
# ---------------------------------------------------------------------------

lab1_page = st.Page("Lab1.py", title="Lab 1", icon="📄")
lab2_page = st.Page("Lab2.py", title="Lab 2", icon="📝")
lab3_page = st.Page("Lab3.py", title="Lab 3", icon="💬", default=True)

pg = st.navigation([lab2_page, lab1_page, lab3_page])
pg.run()