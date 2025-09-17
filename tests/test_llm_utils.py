import unittest

from agentic_qa_assistant.llm_utils import is_reasoning_model, token_params_for_model


class TestLLMUtils(unittest.TestCase):
    def test_is_reasoning_model(self):
        self.assertTrue(is_reasoning_model("gpt-5-nano"))
        self.assertTrue(is_reasoning_model("o4-mini"))
        self.assertFalse(is_reasoning_model("gpt-4o-mini"))
        self.assertFalse(is_reasoning_model("text-embedding-3-small"))

    def test_token_params_for_reasoning_models(self):
        self.assertEqual(token_params_for_model("gpt-5-nano", 123), {"max_completion_tokens": 123})
        self.assertEqual(token_params_for_model("o4-mini", 456), {"max_completion_tokens": 456})

    def test_token_params_for_non_reasoning_models(self):
        self.assertEqual(token_params_for_model("gpt-4o-mini", 200), {"max_tokens": 200})
        self.assertEqual(token_params_for_model("gpt-3.5-turbo", 300), {"max_tokens": 300})

    def test_token_params_none(self):
        self.assertEqual(token_params_for_model("gpt-5-nano", None), {})
        self.assertEqual(token_params_for_model("gpt-4o-mini", None), {})

    def test_temperature_params_and_chat_params(self):
        from agentic_qa_assistant.llm_utils import temperature_params_for_model, chat_params_for_model
        # Reasoning models should omit temperature
        self.assertEqual(temperature_params_for_model("gpt-5-mini", 0.2), {})
        self.assertEqual(chat_params_for_model("gpt-5-mini", 150, temperature=0.2), {"max_completion_tokens": 150})
        # Non-reasoning should include temperature and use max_tokens
        self.assertEqual(temperature_params_for_model("gpt-4o-mini", 0.2), {"temperature": 0.2})
        self.assertEqual(chat_params_for_model("gpt-4o-mini", 150, temperature=0.2), {"max_tokens": 150, "temperature": 0.2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
