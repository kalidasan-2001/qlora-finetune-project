import unittest
from src.training.inference import load_model, generate_response

class TestInference(unittest.TestCase):

    def setUp(self):
        self.model = load_model('path/to/model')  # Adjust the path as necessary

    def test_generate_response(self):
        prompt = "What is the capital of France?"
        expected_response = "The capital of France is Paris."
        response = generate_response(self.model, prompt)
        self.assertEqual(response, expected_response)

if __name__ == '__main__':
    unittest.main()