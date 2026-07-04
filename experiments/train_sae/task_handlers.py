class BaseTaskHandler:
    """Base class for all task handlers."""
    def process_item(self, item, tokenizer):
        """
        Takes a raw dataset item and returns a tuple:
        (prompt_text, positive_answer_text, negative_answer_text)
        
        If an example only has a positive or negative answer, return None for the other.
        """
        raise NotImplementedError("Subclasses must implement process_item()")


class SycophancyHandler(BaseTaskHandler):
    """Handler for Scenario 1: Sycophancy (Anthropic dataset)."""
    def process_item(self, item, tokenizer):
        question = item["question"]
        ans_pos = item["answer_matching_behavior"]
        ans_neg = item["answer_not_matching_behavior"]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt, ans_pos, ans_neg


class IdkHandler(BaseTaskHandler):
    """Handler for Scenario 2: I Don't Know (TriviaQA custom dataset)."""
    def process_item(self, item, tokenizer):
        question = item["question"]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question directly and concisely."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if item["label"] == "unknown":
            # Model doesn't know the answer (Positive behavior we want to steer towards)
            return prompt, item["answer"], None
        else:
            # Model knows the factual answer (Negative behavior)
            return prompt, None, item["answer"]

# ==========================================
# REGISTRY: Add new tasks here in the future!
# ==========================================
TASK_HANDLERS = {
    "sycophancy": SycophancyHandler,
    "idk": IdkHandler
}