# reporting.py
import time
from tqdm import tqdm

class ConsoleLogger:
    """A simple class to handle formatted console output."""
    def __init__(self, verbosity=1):
        self.verbosity = verbosity

    def _log(self, message):
        """Internal log function that uses tqdm.write to avoid breaking progress bars."""
        if self.verbosity > 0:
            # tqdm.write is a static method that prints above a running tqdm bar
            tqdm.write(message)

    def header(self, message):
        """Prints a main header."""
        self._log(f"\n{'='*10} {message.upper()} {'='*10}")

    def info(self, message):
        """Prints a standard informational message."""
        self._log(f"[*] {message}")

    def success(self, message):
        """Prints a success message."""
        self._log(f"[+] {message}")

    def start_timer(self, message="Task"):
        """Starts and returns a simple timer."""
        self.info(f"{message}...")
        return time.time()

    def end_timer(self, start_time):
        """Ends a timer and prints the elapsed time."""
        elapsed = time.time() - start_time
        self.info(f"Completed in {elapsed:.2f}s")