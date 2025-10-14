## FLASK SERVER FOR RESEARCH

Run these commands:

### 1. Create virtual environment inside project root directory
```
python -m venv .venv
```

### 2. Activate virtual environment

```
.\.venv\Scripts\activate
```

### 3. Install requirements

This installs spaCy and English and Tagalog NLP pipelines. Estimated storage size of 1.8 GB, install aswell transformers and torch for the pre-trained library and building deep learning models the size should be around 200mb

```
python -m pip install -r .\requirements.txt
```

### 4. Run the server
```
python app.py
```


## For running docker
```
docker run -p 8080:8080 flask_api_docker_file
```
