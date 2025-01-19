from cryptography.fernet import Fernet
from base64 import b64encode, b64decode
import hashlib
import os
from typing import Optional

class Encryption:
    def __init__(self, key: Optional[str] = None):
        self.key = key or os.getenv('ENCRYPTION_KEY') or Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt a string"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt an encrypted string"""
        try:
            decrypted_data = self.cipher_suite.decrypt(b64decode(encrypted_data))
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password with optional salt"""
        salt = salt or os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            100000
        )
        return b64encode(key).decode(), b64encode(salt).decode() 