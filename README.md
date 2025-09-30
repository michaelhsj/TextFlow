# OCR-Challenge
An end-to-end MLops pipeline project focused on optical character recognition problems. Monitored, fully reproducible, and compliant with auditing needs.

# Directory Overview:
- **ml**: Jupyter notebooks, scripts, and supporting libs for training text extraction models from images.
- **dataset**: .gitignore'd directory for download and processing data during development.
- **model**: .gitignore'd directory for outputting/downloading models for deployment/EDA.
- **service**: FastAPI service to serve inference results over HTTP.
- **deployment**: Terraform scripts and Dockerfiles used to deploy the model and service to GCP.

Details are available in each directory's `README.md`.

# Setup venv
```
python3.11 -m venv .venv-textflow
source .venv-textflow/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
 Update `requirements.txt` with only the necessary packages. Bloat is not free because deployed Docker images will be larger: 
 ```
 pip freeze > requirements.txt
 ```