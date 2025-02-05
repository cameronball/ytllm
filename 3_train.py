import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


class YouTubeTranscriptDataset(Dataset):
    """
    A PyTorch Dataset that loads a CSV file with columns 'url', 'title', and 'transcript'.
    The input to the model is a prompt created by prepending "transcribe: " to the title.
    The target is the transcript.
    """

    def __init__(
        self,
        csv_file,
        tokenizer,
        max_input_length=256,
        max_target_length=1024,
    ):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title = row["title"]
        transcript = row["transcript"]

        # Create a prompt for the model.
        # (You can remove or adjust the "transcribe: " prefix if desired.)
        input_text = "transcribe: " + title
        target_text = transcript

        # Tokenize the inputs and targets
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.squeeze()  # remove batch dimension
        attention_mask = inputs.attention_mask.squeeze()
        labels = targets.input_ids.squeeze()

        # Replace all pad token IDs in labels by -100 so they are ignored by the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TestGenerationCallback(TrainerCallback):
    """
    A custom callback that, at regular intervals, generates a test transcript from a given test title.
    Adjust `test_title` as desired.
    """

    def __init__(self, tokenizer, test_title, eval_interval=500):
        self.tokenizer = tokenizer
        self.test_title = test_title
        self.eval_interval = eval_interval

    def on_step_end(self, args, state, control, **kwargs):
        # Check if the current step is a multiple of the evaluation interval.
        if state.global_step % self.eval_interval == 0 and state.global_step > 0:
            model = kwargs["model"]
            device = next(model.parameters()).device

            # Prepare the prompt using the test title.
            prompt = "transcribe: " + self.test_title
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            # Generate output (you can adjust generation parameters as needed)
            outputs = model.generate(**inputs, max_length=512, num_beams=4)
            transcript = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"\n=== Test Output at step {state.global_step} ===")
            print(f"Title: {self.test_title}")
            print("Generated Transcript:")
            print(transcript)
        return control


def main():
    # === Configuration ===
    csv_file = "output.csv"  # path to your CSV file with columns "url,title,transcript"
    model_name = "t5-small"  # or replace with another T5 variant or seq2seq model
    output_dir = "./output"
    num_train_epochs = 10
    per_device_train_batch_size = 16  # adjust according to your GPU memory
    per_device_eval_batch_size = 16
    save_steps = 500  # save a checkpoint every 500 training steps
    logging_steps = 100  # log every 100 steps
    eval_interval = 500  # generate a test output every 500 steps
    learning_rate = 5e-6

    # === Initialize tokenizer and model ===
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Enable fp16 (mixed-precision) training for speed on Nvidia GPUs (optional)
    fp16 = True if torch.cuda.is_available() else False

    # === Prepare the dataset ===
    full_dataset = YouTubeTranscriptDataset(csv_file, tokenizer)
    dataset_length = len(full_dataset)
    # Split dataset into 90% training and 10% evaluation
    train_size = int(0.9 * dataset_length)
    val_size = dataset_length - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Loaded {dataset_length} samples: {train_size} for training and {val_size} for evaluation.")

    # === Training arguments ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_steps=save_steps,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=save_steps,  # evaluate as often as saving checkpoints
        save_total_limit=2,  # only keep the 2 most recent checkpoints
        learning_rate=learning_rate,
        fp16=fp16,
        load_best_model_at_end=False,
    )

    # === Create the Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[TestGenerationCallback(tokenizer, test_title="Will This Superbug Destroy Us All?", eval_interval=eval_interval)],
    )

    # === Check for existing checkpoints to resume training ===
    checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # Resume from the checkpoint with the highest step number.
            checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            print(f"Resuming training from checkpoint: {checkpoint}")

    # === Start training ===
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save the final model (+ tokenizer and config) to the output directory.
    trainer.save_model(output_dir)
    print("Training complete. Model saved to", output_dir)


if __name__ == "__main__":
    main()
