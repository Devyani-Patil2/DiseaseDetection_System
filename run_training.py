"""
Training wrapper script that captures all output to a log file.
"""
import sys
import os
import io

# Set up logging to file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")
log_file = open(log_path, "w", buffering=1)  # line-buffered

class TeeOutput:
    """Write to both stdout and log file."""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = TeeOutput(sys.__stdout__, log_file)
sys.stderr = TeeOutput(sys.__stderr__, log_file)

print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)
print(f"Log file: {log_path}")

try:
    from src.train import train
    model, history, test_acc = train()
    print(f"\n\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print("TRAINING COMPLETED SUCCESSFULLY!")
except Exception as e:
    print(f"\nERROR DURING TRAINING: {e}")
    import traceback
    traceback.print_exc()
finally:
    log_file.close()
