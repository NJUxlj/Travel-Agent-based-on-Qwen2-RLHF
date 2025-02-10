import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import Agent



def populate_envs(sender_email, receiver_email, subject):
    os.environ['FROM_EMAIL'] = sender_email
    os.environ['TO_EMAIL'] = receiver_email
    os.environ['EMAIL_SUBJECT'] = subject
    
    
    



def send_email(sender_email, receiver_email, subject, thread_id):
    try:
        populate_envs(sender_email, receiver_email, subject)
        config = {"configurable": {"thread_id": thread_id}}
    
    except Exception as e:
        st.error(f"Error sending email: {e}")
    
    



def render_custom_css():
    st.markdown(
        
    )



