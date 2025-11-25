import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
from dotenv import load_dotenv
import json


try:
    load_dotenv()
    FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

    if FIREBASE_CREDENTIALS is None:
        raise ValueError("Credentials is not found")
except Exception as e:
    print(f"Error Loading Credentials: {e}")


class DocumentLogs:

    def __init__(self):

        try:
            cred_dict = json.loads(FIREBASE_CREDENTIALS)
        except Exception as e:
            raise ValueError(f"Invalid FIREBASE_CREDENTIALS JSON: {e}")

        self.cred = credentials.Certificate(cred_dict)

        # initialize firebase app if not already done
        if not firebase_admin._apps:
            firebase_admin.initialize_app(self.cred)

        # create firestore client
        self.db = firestore.client()

    def paradise_logs(self, input, valid, searches ):
        doc_ref = self.db.collection('love-in-paradise-logs').document()
        data = {
            'input': input,
            'searches': searches,
            'valid cliam': valid,
            'log id': doc_ref.id
        }
        print("Writing to Firestore...")
        doc_ref.set(data)
        print("Successfully written:", doc_ref.id)

    def user_input(self, user_input):
        return user_input

    def valid_claim(self, valid_claim):
        return valid_claim

    def search_log(self, search_log):
        print("testing")
        return search_log
