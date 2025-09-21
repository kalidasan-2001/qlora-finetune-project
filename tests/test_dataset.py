import unittest
from src.training.dataset import DatasetLoader

class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.dataset_loader = DatasetLoader("path/to/test_dataset.jsonl")

    def test_load_data(self):
        data = self.dataset_loader.load_data()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_data_transformation(self):
        raw_data = [{"input": "test input", "output": "expected output"}]
        transformed_data = self.dataset_loader.transform_data(raw_data)
        self.assertEqual(len(transformed_data), 1)
        self.assertIn("input_ids", transformed_data[0])
        self.assertIn("attention_mask", transformed_data[0])

if __name__ == '__main__':
    unittest.main()