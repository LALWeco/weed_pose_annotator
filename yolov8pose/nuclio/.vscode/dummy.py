import yaml
import json
with open("function.yaml", "rb") as function_file:
    functionconfig = yaml.safe_load(function_file)
    labels_spec = functionconfig["metadata"]["annotations"]["spec"]
    labels = json.loads(labels_spec)