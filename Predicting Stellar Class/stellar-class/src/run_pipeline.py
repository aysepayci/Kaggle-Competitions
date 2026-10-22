from pathlib import Path
import subprocess, sys, textwrap

base = Path(__file__).resolve().parent.parent
explore = base / 'src' / 'explore_data.py'
train = base / 'src' / 'train_model.py'

print('Running exploratory inspection...')
subprocess.run([sys.executable, str(explore)], check=True)
print('Running model training...')
subprocess.run([sys.executable, str(train)], check=True)
print('Pipeline finished. Submission generated in', (base / 'submissions').resolve())
