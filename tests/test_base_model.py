import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifttrack.models import User, UserUpdate

def test_user_model():
    user = User(fname="John", lname="Doe", username="johndoe", phoneNum="1234567890", email="johndoe@example.com", password="password")
    print(user.model_dump(mode='python'))
    
def test_user_update_model():
    user_update = UserUpdate(email="johndoe@example.com")
    print(user_update.model_dump(mode='python', exclude_none=True))
    
if __name__ == "__main__":
    test_user_model()
    test_user_update_model()