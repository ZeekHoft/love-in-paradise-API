import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentLogs:

    def __init__(self):
        FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

        if FIREBASE_CREDENTIALS is None:
            raise ValueError("FIREBASE_CREDENTIALS is not found in environment variables")

        # Convert JSON string → dict
        try:
            cred_dict = json.loads(FIREBASE_CREDENTIALS)
        except Exception as e:
            raise ValueError(f"Invalid FIREBASE_CREDENTIALS JSON: {e}")

        cred = credentials.Certificate(cred_dict)

        # initialize firebase app once
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()

    def paradise_logs(self, input, valid, searches):
        doc_ref = self.db.collection('love-in-paradise-logs').document()
        data = {
            'input': input,
            'searches': searches,
            'valid claim': valid,
            'log id': doc_ref.id
        }
        print("Writing to Firestore...")
        doc_ref.set(data)
        print("Successfully written:", doc_ref.id)


# logs = DocumentLogs()
# print(logs.paradise_logs(logs.tokens()))


# firebase_admin.initialize_app(cred)


# db = firestore.client()
# doc_ref = db.collection('love-in-paradise-logs').document()

# data = {
#     'logs': 'wash hands',
#     'log id': doc_ref.id
    
# }
# doc_ref.set(data)
