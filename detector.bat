set original_dir = %CD%
set venv_root_dir = "F:\python\aia_git\ai_annotator\venv"

call "F:\python\aia_git\ai_annotator\venv"\Scripts\activate.bat

call F:\python\aia_git\ai_annotator\venv\Scripts\python  F:\python\aia_git\ai_annotator\detector.py

call "F:\python\aia_git\ai_annotator\venv"\Scripts\deactivate.bat

