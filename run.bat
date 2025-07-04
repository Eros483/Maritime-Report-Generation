@echo off
echo Activating conda environment and running Streamlit app...

call "C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat" base

:: Activate your specific environment
call conda activate rag_new_env

:: Run streamlit app
streamlit run frontend.py

:: Keep window open if there's an error
pause