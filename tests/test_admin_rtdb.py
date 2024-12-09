import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lifttrack.v2.dbhelper.admin_rtdb import FirebaseDBHelper


class TestFirebaseDBHelper(unittest.TestCase):
    @patch('firebase_admin.credentials.Certificate')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin.firestore.client')
    def setUp(self, mock_client, mock_init, mock_cert):
        """Set up test fixtures before each test method"""
        self.mock_db = MagicMock()
        mock_client.return_value = self.mock_db
        self.db_helper = FirebaseDBHelper(credentials_path="fake_path")

    # def tearDown(self):
    #     """Clean up after each test method"""
    #     self.db_helper.close()

    # [PASSED]
    # def test_a_singleton_pattern(self):
    #     """Test that FirebaseDBHelper maintains singleton pattern"""
    #     instance1 = FirebaseDBHelper(credentials_path="fake_path")
    #     instance2 = FirebaseDBHelper(credentials_path="fake_path")
    #     self.assertIs(instance1, instance2)

    # [PASSED]
    def test_b_add_document(self):
        """Test adding a document with and without custom ID"""
        # Test with auto-generated ID
        collection_mock = MagicMock()
        self.mock_db.collection.return_value = collection_mock
        doc_ref = MagicMock()
        collection_mock.add.return_value = (None, doc_ref)
        doc_ref.id = "auto_generated_id"

        data = {"field": "value"}
        result = self.db_helper.add_document("test_collection", data)
        print("Expected:", doc_ref.id, "Actual:", result)
        self.assertEqual(result, "auto_generated_id")
        collection_mock.add.assert_called_once_with(data)

        # Test with custom ID
        doc_mock = MagicMock()
        collection_mock.document.return_value = doc_mock

        custom_id = "custom_id"
        result = self.db_helper.add_document("test_collection", data, doc_id=custom_id)
        self.assertEqual(result, custom_id)
        doc_mock.set.assert_called_once_with(data)

    def test_c_get_document(self):
        """Test retrieving a document"""
        collection_mock = MagicMock()
        self.mock_db.collection.return_value = collection_mock
        doc_mock = MagicMock()
        collection_mock.document.return_value = doc_mock

        # Mock document exists
        doc_snapshot = MagicMock()
        doc_snapshot.exists = True
        doc_snapshot.to_dict.return_value = {"field": "value"}
        doc_mock.get.return_value = doc_snapshot

        result = self.db_helper.get_document("test_collection", "doc_id")
        self.assertEqual(result, {"field": "value"})

        # Mock document doesn't exist
        doc_snapshot.exists = False
        result = self.db_helper.get_document("test_collection", "doc_id")
        self.assertIsNone(result)

    def test_d_update_document(self):
        """Test updating a document"""
        collection_mock = MagicMock()
        self.mock_db.collection.return_value = collection_mock
        doc_mock = MagicMock()
        collection_mock.document.return_value = doc_mock

        update_data = {"field": "new_value"}
        result = self.db_helper.update_document("test_collection", "doc_id", update_data)
        self.assertTrue(result)
        doc_mock.update.assert_called_once_with(update_data)

    def test_e_delete_document(self):
        """Test deleting a document"""
        collection_mock = MagicMock()
        self.mock_db.collection.return_value = collection_mock
        doc_mock = MagicMock()
        collection_mock.document.return_value = doc_mock

        result = self.db_helper.delete_document("test_collection", "doc_id")
        self.assertTrue(result)
        doc_mock.delete.assert_called_once()

    def test_f_query_collection(self):
        """Test querying a collection with filters"""
        collection_mock = MagicMock()
        self.mock_db.collection.return_value = collection_mock

        # Mock query results
        doc1 = MagicMock()
        doc1.to_dict.return_value = {"field": "value1"}
        doc2 = MagicMock()
        doc2.to_dict.return_value = {"field": "value2"}

        query_mock = MagicMock()
        query_mock.stream.return_value = [doc1, doc2]

        # Setup query chain
        collection_mock.where.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.limit.return_value = query_mock

        filters = [("field", "==", "value")]
        result = self.db_helper.query_collection(
            "test_collection",
            filters=filters,
            order_by="field",
            limit=2
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"field": "value1"})
        self.assertEqual(result[1], {"field": "value2"})

    def test_g_batch_write(self):
        """Test batch write operations"""
        batch_mock = MagicMock()
        self.mock_db.batch.return_value = batch_mock

        def operation1(batch):
            batch.set("doc1", {"field": "value1"})

        def operation2(batch):
            batch.set("doc2", {"field": "value2"})

        operations = [operation1, operation2]
        result = self.db_helper.batch_write(operations)

        self.assertTrue(result)
        batch_mock.commit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
