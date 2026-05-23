# Al Gharafa Hidden Gems Scouting Tool - Deploy Folder

Upload this folder to GitHub, then deploy it on Streamlit Community Cloud.

Required root files:
- app.py
- requirements.txt
- data/strikers.csv
- data/dna_by_season.csv
- data/formation_summary.csv
- data/formation_by_season.csv
- assets/al_gharafa_logo.png

Streamlit settings:
- Repository: your GitHub repo
- Branch: main
- Main file path: app.py

Local run:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
