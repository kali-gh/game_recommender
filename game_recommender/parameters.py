import logging
import json


logger = logging.getLogger(__name__)


class Params:
    def __init__(self, file_params):
        with open(file_params, 'r') as f:
            self._params  = json.load(f)
        logger.info(f"Loaded params from {file_params}")
        logger.info(str(self))

    def __getitem__(self, item):
        if item not in self._params.keys():
            raise AttributeError(f"Attribute '{item}' not found in params.")
        else:
            return self._params.get(item, None)

    def __str__(self):
        return json.dumps(self._params, indent=4, sort_keys=True)