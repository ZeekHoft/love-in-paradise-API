from stanza.server import CoreNLPClient
from stanza.server.client import StartServer
from stanza.server.client import CoreNLPClient
from pathlib import Path
import subprocess
import requests
import time
import os

"""
python
>>> import stanza
>>> stanza.install_corenlp()

OR 

python -c "import stanza;stanza.install_corenlp()"
"""

# Get the relative filepath of property file
# Should be forward slash instead of backslash provided by os.path
props_filepath = os.path.relpath(
    os.path.join(os.path.dirname(__file__), "corenlp.props")
).replace("\\", "/")
classpath = os.getenv("CORENLP_HOME", os.path.join(str(Path.home()), "stanza_corenlp"))
if classpath:
    classpath = os.path.join(classpath, "*")


class OpenInformationExtraction:
    """
    Uses Stanza/Stanford OpenIE for extracting relations from text sentences.
    Starts or connects to a local CoreNLP server.
    """

    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.client = None
        self._server_proc = None
        self._connect_with_retry()

    def _start_server_suppressed(self, port=9000, timeout=15000):
        # Start server with suppressed output
        cmd = [
            "java",
            "-mx4g",
            "-cp",
            str(classpath),
            "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
            "-port",
            str(port),
            "-timeout",
            str(timeout),
        ]
        devnull = subprocess.DEVNULL
        proc = subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
        self._server_proc = proc
        return proc

    def _connect_with_retry(self):
        """Connect to CoreNLP server with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if self.client is None:
                    # First check if server is accessible
                    server_up = self._check_connection()

                    if not server_up:
                        print("CoreNLP server server is down or unavailable")
                        # Start server manually
                        try:
                            self._start_server_suppressed()
                            print("Server started with suppressed output")
                            time.sleep(2 + attempt * 2)
                            self._attempt_connect()
                        except Exception as e:
                            print("Error attempting to connect to suppressed server")
                            pass

                        # Fallback manual start with CoreNLPClient
                        if attempt > 0:
                            try:
                                self.client = CoreNLPClient(
                                    start_server=StartServer.TRY_START,
                                    endpoint="http://localhost:9000",
                                    timeout=60000,
                                    annotators=["openie", "coref"],
                                    be_quiet=True,
                                    properties=props_filepath,
                                )
                                print(
                                    "CoreNLP client connected successfully via TRY START"
                                )
                            except Exception:
                                self.client = None

                    else:
                        print("CoreNLP server is available")
                        self._attempt_connect()
                return

            except Exception as e:
                print(f"Error connecting to CoreNLP: {e}")
                print(
                    f"CoreNLP connection attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Failed to connect to CoreNLP after all retries")
                    raise RuntimeError("CoreNLP server unavailable")

    def _check_connection(self):
        """Check if CoreNLP is still accessible"""
        try:
            response = requests.get("http://localhost:9000/", timeout=1)
            return response.status_code == 200
        except:
            return False

    def _attempt_connect(self):
        print("Attempting to connect CoreNLP client...")
        # Connect to server without asking stanza to start its own process
        self.client = CoreNLPClient(
            start_server=StartServer.DONT_START,
            endpoint="http://localhost:9000",
            timeout=60000,
            annotators=["openie", "coref"],
            be_quiet=True,
            properties=props_filepath,
        )
        print("CoreNLP client connected successfully to existing server")

    def generate_triples(self, text, retry_on_failure=True):
        """
        Extract triples from text using OpenIE with automatic retry

        Args:
            text (str): Input text to extract triples from
            retry_on_failure (bool): Whether to retry on connection failures

        Returns:
            list: List of tuples (subject, relation, object)
        """
        if not text or not text.strip():
            return []

        for attempt in range(self.max_retries if retry_on_failure else 1):
            try:
                # Check if we need to reconnect
                if self.client is None:
                    print("Client disconnected, attempting to reconnect...")
                    self._connect_with_retry()

                # Annotate text
                ann = self.client.annotate(text)

                # Returns a triple for each sentence
                triples = []

                for sentence in ann.sentence:
                    for triple in sentence.openieTriple:
                        subject = triple.subject.strip()
                        relation = triple.relation.strip()
                        obj = triple.object.strip()

                        if subject and relation and obj:
                            triples.append((subject, relation, obj))

                return triples

            except Exception as e:
                error_msg = str(e)
                is_connection_error = (
                    "Connection refused" in error_msg
                    or "Max retries exceeded" in error_msg
                    or "Failed to establish" in error_msg
                    or "Connection reset" in error_msg
                )

                if (
                    is_connection_error
                    and retry_on_failure
                    and attempt < self.max_retries - 1
                ):
                    print(
                        f"CoreNLP connection lost (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )

                    # Mark client as disconnected
                    self.client = None

                    # Check if server is still up
                    if not self._check_connection():
                        print("CoreNLP server appears to be down")
                        wait_time = (attempt + 1) * 3
                        print(f"Waiting {wait_time}s for server recovery...")
                        time.sleep(wait_time)
                    else:
                        # Server is up but connection failed, try reconnecting
                        time.sleep(1)
                        try:
                            self._connect_with_retry()
                        except:
                            print("Reconnection failed, will retry on next attempt")
                else:
                    # Non-connection error or max retries reached
                    print(f"Error generating triples: {e}")
                    return []

        # All retries exhausted
        print(f"Failed to generate triples after {self.max_retries} attempts")
        return []

    def __del__(self):
        """Cleanup: terminate server and close client"""
        try:
            if self.client is not None:
                self.client.close()
        except:
            pass
        try:
            if self._server_proc is not None:
                self._server_proc.terminate()
        except:
            pass
