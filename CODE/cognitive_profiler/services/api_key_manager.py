import logging

class SimpleApiKeyManager:
    """[REDACTED_BY_SCRIPT]"""
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        self.keys = keys
        self.current_key_index = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[REDACTED_BY_SCRIPT]")

    def get_key(self) -> str:
        return self.keys[self.current_key_index]

    def rotate_key(self) -> str:
        """[REDACTED_BY_SCRIPT]"""
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        self.logger.warning(f"[REDACTED_BY_SCRIPT]")
        return self.get_key()