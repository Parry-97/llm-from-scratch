import tiktoken
import time
from torch.utils.data import DataLoader
import torch
from llm_from_scratch.gpt_architecture.text_generation import generate_text_simple
from llm_from_scratch.clf_finetuning.spam_dataset import SpamDataset
from llm_from_scratch.gpt_download import download_and_load_gpt2
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from llm_from_scratch.pretraining.utils import (
    load_weights_into_gpt,
    text_to_token,
    token_ids_to_text,
)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # NOTE: Sets the model to evaluation mode to disable dropout
    with torch.no_grad():  # NOTE: Disables gradient computation to reduce overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# INFO: To determine the classification accuracy, we apply the argmax-based prediction
# code to all examples in the dataset and calculate the proportion of correct predictions
# by defining a calc_accuracy_loader function.
def calc_accuracy_loader(
    data_loader: DataLoader, model: GPTModel, device, num_batches=None
):
    # Disable dropout and batchnorm layers
    model.eval()

    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # logits of the last token
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]

            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


# NOTE: Before we begin fine-tuning the model, we must define the loss function
# we will optimize during training. Our objective is to maximize the spam classification
# accuracy of the model, which means that the preceding code should output the cor-
# rect class labels: 0 for non-spam and 1 for spam.
# Because classification accuracy is not a differentiable function, we use cross-
# entropy loss as a proxy to maximize accuracy.


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # logits of the last token
    batch_loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return batch_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)

load_weights_into_gpt(model, params)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token(text_1, tokenizer=tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"],
)

print(token_ids_to_text(token_ids, tokenizer=tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"],
)
print(token_ids_to_text(token_ids, tokenizer))

# NOTE: To get the model ready for classification fine-tuning,
# we first freeze the model, meaning that we make all layers nontrainable:

for param in model.parameters():
    param.requires_grad = False

# NOTE: Then, we replace the output layer (model.out_head), which originally maps the layer
# inputs to 50,257 dimensions, the size of the vocabulary
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
)

# INFO: fine-tuning additional layers can noticeably improve the predictive performance of the model.
# We also configure the last transformer block and the final LayerNorm
# module, which connects this block to the output layer, to be trainable,
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)

print(f"Inputs: {inputs}")
print(f"Input dimensions: {inputs.shape}")

train_dataset = SpamDataset(
    file_path="tests/train.csv", max_length=None, tokenizer=tokenizer
)

val_dataset = SpamDataset(
    file_path="tests/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
)

test_dataset = SpamDataset(
    file_path="tests/test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)


# NOTE: The classifier training loop is similar to the one we used to train the GPT model.
# The only two distinctions are that we now track the number of training examples seen (examples_seen)
# instead of the number of tokens, and we calculate the accuracy after each epoch instead
# of printing a sample text.


def train_classifier(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,  # pyright: ignore
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_steps: int,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, 0

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # reset gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # compute gradients

            optimizer.step()  # update parameters
            examples_seen += input_batch.shape[
                0
            ]  # tracks examples seen instead of tokens
            global_step += 1

            if global_step % eval_freq == 0:
                # NOTE: The evaluate_model function is the same as the one we used for pretraining
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_steps
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f" Epoch: {epoch + 1} | Global Step: {global_step:06d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

        train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_steps)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_steps)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


start = time.time()
torch.manual_seed(123)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)  # pyright: ignore
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=torch.device("cpu"),
    num_epochs=num_epochs,
    eval_freq=50,
    eval_steps=5,
)

end = time.time()
execution_time_minutes = (end - start) / 60

print(f"Execution time: {execution_time_minutes:.2f} minutes")
