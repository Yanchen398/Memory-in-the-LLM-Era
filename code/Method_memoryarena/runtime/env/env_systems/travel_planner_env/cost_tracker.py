OPENAI_PRICING = {
    'gpt-5.1': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5.1-chat-latest': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5.1-codex-max': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5.1-codex': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'cached_input': 0.025, 'output': 2.00},
    'gpt-5-nano': {'input': 0.05, 'cached_input': 0.005, 'output': 0.40},
    'gpt-5-chat-latest': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5-codex': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
    'gpt-5-pro': {'input': 15.00, 'cached_input': None, 'output': 120.00},
    'gpt-4.1': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
    'gpt-4.1-mini': {'input': 0.40, 'cached_input': 0.10, 'output': 1.60},
    'gpt-4.1-nano': {'input': 0.10, 'cached_input': 0.025, 'output': 0.40},
    'gpt-4o': {'input': 2.50, 'cached_input': 1.25, 'output': 10.00},
    'gpt-4o-2024-05-13': {'input': 5.00, 'cached_input': None, 'output': 15.00},
    'gpt-4o-mini': {'input': 0.15, 'cached_input': 0.075, 'output': 0.60},
    'gpt-4-turbo': {'input': 10.00, 'cached_input': None, 'output': 30.00},
    'gpt-4-1106-preview': {'input': 10.00, 'cached_input': None, 'output': 30.00},
    'gpt-3.5-turbo': {'input': 0.50, 'cached_input': None, 'output': 1.50},
    'default': {'input': 2.50, 'cached_input': None, 'output': 10.00}
}

GEMINI_PRICING = {
    'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
    'gemini-2.0-flash-lite': {'input': 0.075, 'output': 0.30},
    'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
    'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
    'gemini-1.5-flash-8b': {'input': 0.0375, 'output': 0.15},
    'gemini-pro': {'input': 0.50, 'output': 1.50},
    'default': {'input': 0.50, 'output': 1.50}
}

OTHER_PRICING = {
    'claude-3-opus': {'input': 15.00, 'output': 75.00},
    'claude-3-sonnet': {'input': 3.00, 'output': 15.00},
    'claude-3-haiku': {'input': 0.25, 'output': 1.25},
    'default': {'input': 2.50, 'output': 10.00}
}


class CostTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = self._detect_provider(model_name)
        self.total_input_tokens = 0
        self.total_cached_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _detect_provider(self, model_name: str) -> str:
        model_lower = model_name.lower()
        if 'gemini' in model_lower:
            return 'gemini'
        elif 'claude' in model_lower:
            return 'anthropic'
        elif 'gpt' in model_lower or 'o1' in model_lower:
            return 'openai'
        else:
            return 'other'

    def add_usage(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens
        self.call_count += 1

    def get_pricing(self) -> dict:
        if self.provider == 'openai':
            return OPENAI_PRICING.get(self.model_name, OPENAI_PRICING['default'])
        elif self.provider == 'gemini':
            return GEMINI_PRICING.get(self.model_name, GEMINI_PRICING['default'])
        else:
            return OTHER_PRICING.get(self.model_name, OTHER_PRICING['default'])

    def get_cost(self) -> dict:
        pricing = self.get_pricing()
        input_cost = (self.total_input_tokens / 1_000_000) * pricing['input']
        output_cost = (self.total_output_tokens / 1_000_000) * pricing['output']
        cached_cost = 0
        if pricing.get('cached_input') and self.total_cached_tokens > 0:
            cached_cost = (self.total_cached_tokens / 1_000_000) * pricing['cached_input']
        return {
            'model': self.model_name,
            'provider': self.provider,
            'call_count': self.call_count,
            'input_tokens': self.total_input_tokens,
            'cached_tokens': self.total_cached_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_cached_tokens + self.total_output_tokens,
            'input_cost': input_cost,
            'cached_cost': cached_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + cached_cost + output_cost
        }

    def reset(self):
        self.total_input_tokens = 0
        self.total_cached_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0