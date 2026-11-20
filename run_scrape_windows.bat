@echo off
python -m pip install -r requirements.txt
python scrape.py --out data\full_dataset --max-pages 20000 --delay 1.5
pause
