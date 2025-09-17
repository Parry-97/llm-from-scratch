from torch.utils.data import Dataset
import tiktoken
from .utils import format_input


class InstructionDataset(Dataset):
    """Pytorch dataset for instruction fine-tuning."""

    def __init__(self, data: dict, tokenizer: tiktoken.Encoding):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.encoded_texts)
