import streamlit as st
import pandas as pd
import yfinance as yf

st.write(
    """ 
# Chat app using AnnotatorLLM\n
  Give an entity and the bot will give you his wikidata entity
"""
)


tickerSymbol = "GOOGL"
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(
    period="id", start="2010-05-31", end="2024-01-31")

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)
