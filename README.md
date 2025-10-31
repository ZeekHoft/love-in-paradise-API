## FLASK SERVER API

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

This installs Python dependencies and NLP models. Estimated storage size of 1.8 GB. Installs aswell transformers and torch for the pre-trained library and building deep learning models the size should be around 200mb.

```
python -m pip install -r .\requirements.txt
```

### 4. Install CoreNLP

Installs the Stanford CoreNLP Server. Make sure Java is already installed in your system.

```
python -c "import stanza;stanza.install_corenlp()"
```

### 5. Running directly

Run the algorithm in Python to see if it works with no error.

```
python main.py
```

### 6. Run as a Flask server

Starts an API server on localhost port 8080.

```
python app.py
```

## FOR RUNNING DOCKER

```
docker run -p 8080:8080 flask_api_docker_file
```
