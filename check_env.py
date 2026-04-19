import sys
import json

info = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": sys.platform,
}

try:
    import tensorflow as tf
    info["tf_version"] = tf.__version__
    info["tf_ok"] = True
    gpus = tf.config.list_physical_devices('GPU')
    info["gpus"] = [str(g) for g in gpus]
except Exception as e:
    info["tf_ok"] = False
    info["tf_error"] = str(e)

try:
    import numpy as np
    info["numpy_version"] = np.__version__
except:
    info["numpy_ok"] = False

try:
    import sklearn
    info["sklearn_version"] = sklearn.__version__
except:
    pass

with open("env_check.json", "w") as f:
    json.dump(info, f, indent=2)

print(json.dumps(info, indent=2))
