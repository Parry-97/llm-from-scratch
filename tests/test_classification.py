"""Tests for GPT model classification fine-tuning."""

import torch
import pytest
from torch.utils.data import DataLoader

from llm_from_scratch.clf_finetuning.spam_dataset import SpamDataset
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from llm_from_scratch.gpt_architecture.text_generation import generate_text_simple
from llm_from_scratch.gpt_download import download_and_load_gpt2
from llm_from_scratch.pretraining.utils import (
    load_weights_into_gpt,
    text_to_token,
    token_ids_to_text,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def classification_config():
    """Configuration for classification model."""
    return {
        "vocab_size": 50257,
        "context_length": 256,  # Reduced for testing
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


@pytest.fixture
def pretrained_model(classification_config):
    """Load a pretrained GPT2 model."""
    torch.manual_seed(123)
    model = GPTModel(classification_config)

    # For testing, we'll use a smaller model configuration
    model_size = "124M"
    try:
        settings, params = download_and_load_gpt2(
            model_size=model_size, models_dir="gpt2"
        )
        load_weights_into_gpt(model, params)
    except Exception:
        # If download fails, continue with random weights for testing
        pass

    return model


@pytest.fixture
def classification_model(pretrained_model, classification_config):
    """Create a classification model from pretrained GPT."""
    torch.manual_seed(123)
    num_classes = 2

    # NOTE: To get the model ready for classification fine-tuning,
    # we first freeze the model, meaning that we make all layers nontrainable:
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # NOTE: Then, we replace the output layer (model.out_head), which originally maps the layer
    # inputs to 50,257 dimensions, the size of the vocabulary
    pretrained_model.out_head = torch.nn.Linear(
        in_features=classification_config["emb_dim"], out_features=num_classes
    )

    # INFO: fine-tuning additional layers can noticeably improve the predictive performance of the model.
    # We also configure the last transformer block and the final LayerNorm
    # module, which connects this block to the output layer, to be trainable,
    for param in pretrained_model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in pretrained_model.final_norm.parameters():
        param.requires_grad = True

    return pretrained_model


@pytest.fixture
def spam_train_dataset(tokenizer, tmp_path):
    """Create a small training dataset for testing."""
    # Create a small CSV file for testing
    import pandas as pd

    data = pd.DataFrame(
        {
            "Label": ["ham", "spam", "ham", "spam"],
            "Text": [
                "Hello, how are you?",
                "Win money now! Click here!",
                "Meeting at 3pm tomorrow",
                "Congratulations! You won $1000!",
            ],
        }
    )
    train_file = tmp_path / "train.csv"
    data.to_csv(train_file, index=False)

    return SpamDataset(file_path=str(train_file), max_length=50, tokenizer=tokenizer)


@pytest.fixture
def spam_val_dataset(tokenizer, spam_train_dataset, tmp_path):
    """Create a small validation dataset for testing."""
    import pandas as pd

    data = pd.DataFrame(
        {
            "Label": ["ham", "spam"],
            "Text": ["See you at the office", "Free prize! Act now!"],
        }
    )
    val_file = tmp_path / "validation.csv"
    data.to_csv(val_file, index=False)

    return SpamDataset(
        file_path=str(val_file),
        max_length=spam_train_dataset.max_length,
        tokenizer=tokenizer,
    )


@pytest.fixture
def train_loader(spam_train_dataset):
    """Create a DataLoader for training."""
    batch_size = 2
    num_workers = 0
    torch.manual_seed(123)

    return DataLoader(
        dataset=spam_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )


@pytest.fixture
def val_loader(spam_val_dataset):
    """Create a DataLoader for validation."""
    batch_size = 2
    num_workers = 0

    return DataLoader(
        dataset=spam_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


@pytest.fixture
def device():
    """Get the device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def optimizer(classification_model):
    """Create an optimizer for the classification model."""
    # NOTE: AdamW is a variant of Adam that
    # improves the weight decay approach, which aims to minimize model complexity and
    # prevent overfitting by penalizing larger weights. This adjustment allows AdamW to
    # achieve more effective regularization and better generalization; thus, AdamW is fre-
    # quently used in the training of LLMs.
    return torch.optim.Adam(
        classification_model.parameters(), lr=1e-5, weight_decay=0.1
    )


# ============================================================================
# Helper Functions
# ============================================================================


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets."""
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
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Calculate classification accuracy over a dataloader."""
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

    return correct_predictions / num_examples if num_examples > 0 else 0.0


# NOTE: Before we begin fine-tuning the model, we must define the loss function
# we will optimize during training. Our objective is to maximize the spam classification
# accuracy of the model, which means that the preceding code should output the cor-
# rect class labels: 0 for non-spam and 1 for spam.
# Because classification accuracy is not a differentiable function, we use cross-
# entropy loss as a proxy to maximize accuracy.
def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # logits of the last token
    batch_loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return batch_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over a dataloader."""
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


# NOTE: The classifier training loop is similar to the one we used to train the GPT model.
# The only two distinctions are that we now track the number of training examples seen (examples_seen)
# instead of the number of tokens, and we calculate the accuracy after each epoch instead
# of printing a sample text.
def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_steps,
):
    """Train the classifier model."""
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

        train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_steps)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_steps)

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# ============================================================================
# Test Functions
# ============================================================================


def test_classification_model_initialization(classification_model):
    """Test that classification model is properly initialized."""
    # Check that output head has correct dimensions
    assert classification_model.out_head.out_features == 2

    # Check that some parameters are trainable
    trainable_params = sum(
        1 for p in classification_model.parameters() if p.requires_grad
    )
    assert trainable_params > 0

    # Check that not all parameters are trainable (model should be partially frozen)
    total_params = sum(1 for p in classification_model.parameters())
    assert trainable_params < total_params


def test_spam_dataset_loading(spam_train_dataset, spam_val_dataset):
    """Test that spam datasets load correctly."""
    assert len(spam_train_dataset) > 0
    assert len(spam_val_dataset) > 0

    # Check that dataset returns correct types
    text_tensor, label = spam_train_dataset[0]
    assert isinstance(text_tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.item() in [0, 1]


def test_dataloader_batching(train_loader, val_loader):
    """Test that dataloaders produce correct batch shapes."""
    train_batch = next(iter(train_loader))
    assert len(train_batch) == 2  # input and target
    assert train_batch[0].ndim == 2  # [batch_size, seq_len]
    assert train_batch[1].ndim == 1  # [batch_size]

    val_batch = next(iter(val_loader))
    assert len(val_batch) == 2
    assert val_batch[0].ndim == 2
    assert val_batch[1].ndim == 1


def test_forward_pass(classification_model, train_loader, device):
    """Test that forward pass produces correct output shape."""
    classification_model.to(device)
    classification_model.eval()

    batch = next(iter(train_loader))
    input_batch, target_batch = batch
    input_batch = input_batch.to(device)

    with torch.no_grad():
        logits = classification_model(input_batch)

    # Check output shape
    assert logits.ndim == 3  # [batch_size, seq_len, num_classes]
    assert logits.shape[0] == input_batch.shape[0]

    # Check last token logits
    last_logits = logits[:, -1, :]
    assert last_logits.shape[0] == input_batch.shape[0]
    assert last_logits.shape[1] == 2  # num_classes


def test_loss_calculation(classification_model, train_loader, device):
    """Test that loss calculation works correctly."""
    classification_model.to(device)

    batch = next(iter(train_loader))
    input_batch, target_batch = batch

    loss = calc_loss_batch(input_batch, target_batch, classification_model, device)

    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_accuracy_calculation(classification_model, val_loader, device):
    """Test that accuracy calculation works correctly."""
    classification_model.to(device)

    accuracy = calc_accuracy_loader(val_loader, classification_model, device)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_training_step(classification_model, train_loader, device, optimizer):
    """Test that a training step updates model parameters."""
    torch.manual_seed(123)
    classification_model.to(device)
    classification_model.train()

    # Get initial parameters
    initial_params = [
        p.clone().detach() for p in classification_model.parameters() if p.requires_grad
    ]

    # Perform training step
    batch = next(iter(train_loader))
    input_batch, target_batch = batch

    optimizer.zero_grad()
    loss = calc_loss_batch(input_batch, target_batch, classification_model, device)
    loss.backward()
    optimizer.step()

    # Check that parameters have been updated
    updated_params = [
        p.clone().detach() for p in classification_model.parameters() if p.requires_grad
    ]

    params_changed = any(
        not torch.allclose(initial, updated)
        for initial, updated in zip(initial_params, updated_params)
    )
    assert params_changed, "Model parameters should be updated after training step"


def test_text_generation_simple(classification_model, tokenizer):
    """Test basic text generation with the classification model."""
    classification_model.eval()

    text = "Every effort moves you"
    token_ids = generate_text_simple(
        model=classification_model,
        idx=text_to_token(text, tokenizer),
        max_new_tokens=5,
        context_size=classification_model.pos_emb.weight.shape[0],
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    assert isinstance(generated_text, str)
    assert len(generated_text) > len(text)


def test_spam_classification_prompt(classification_model, tokenizer):
    """Test model response to spam classification prompt."""
    classification_model.eval()

    text = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate_text_simple(
        model=classification_model,
        idx=text_to_token(text, tokenizer),
        max_new_tokens=10,
        context_size=classification_model.pos_emb.weight.shape[0],
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    assert isinstance(generated_text, str)
    assert generated_text.startswith("Is the following text")


def test_input_encoding(tokenizer):
    """Test that tokenizer correctly encodes input."""
    text = "Do you have time"
    inputs = tokenizer.encode(text)
    inputs_tensor = torch.tensor(inputs).unsqueeze(0)

    assert inputs_tensor.ndim == 2
    assert inputs_tensor.shape[0] == 1
    assert inputs_tensor.shape[1] > 0


@pytest.mark.slow
def test_model_evaluation(classification_model, train_loader, val_loader, device):
    """Test model evaluation on train and validation sets."""
    classification_model.to(device)

    train_loss, val_loss = evaluate_model(
        classification_model, train_loader, val_loader, device, eval_iter=1
    )

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert train_loss > 0
    assert val_loss > 0
    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))


@pytest.mark.slow
def test_mini_training_loop(
    classification_model, train_loader, val_loader, device, optimizer
):
    """Test a minimal training loop with a few steps."""
    torch.manual_seed(123)
    classification_model.to(device)

    # Track initial accuracy
    classification_model.eval()
    initial_accuracy = calc_accuracy_loader(train_loader, classification_model, device)

    # Train for a few steps
    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model=classification_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=1,
            eval_freq=2,
            eval_steps=1,
        )
    )

    # Check that training produced some results
    assert len(train_accs) > 0
    assert len(val_accs) > 0
    assert examples_seen > 0

    # Check that accuracies are valid
    for acc in train_accs + val_accs:
        assert 0.0 <= acc <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_training(classification_model, train_loader):
    """Test that model can run on GPU if available."""
    device = torch.device("cuda")
    classification_model.to(device)

    batch = next(iter(train_loader))
    input_batch, target_batch = batch

    loss = calc_loss_batch(input_batch, target_batch, classification_model, device)

    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)

