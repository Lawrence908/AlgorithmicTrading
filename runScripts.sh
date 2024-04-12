#!/bin/bash

#run finalPrototype.py
echo "Running finalPrototype.py"
streamlit run finalPrototype.py &

#run finalMoonShot.py
echo "Running finalMoonShot.py"
streamlit run finalMoonShot.py &

#run finalRecommendations.py
echo "Running finalRecommendations.py"
streamlit run finalRecommendations.py &

#open the Streamlit applications in Chrome
google-chrome --new-tab "http://localhost:8501/" --new-tab "http://localhost:8502/" --new-tab "http://localhost:8503/"


echo "All scripts have finished execution."
