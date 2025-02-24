import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from copy import deepcopy
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, upload_file


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

wandb.login(key=WANDB_API_KEY)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int = 10,
        hf_repo_name: str | None = None,
        experiment_name: str = "default_experiment",
        project_name: str = "clip_finetuning",
        max_iter: int | None = None,
        log_interval: int = 1
    ):
        """
        Trainer class for CLIP fine-tuning on labeled dataset with WandB logging and Hugging Face model storage
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.hf_repo_name = hf_repo_name
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.best_model_state = None
        self.best_val_accuracy = 0.0
        self.max_iter = max_iter
        self.log_interval = log_interval

        train_params, total_params = self.count_parameters()
        wandb.init(project=self.project_name, 
                   name=self.experiment_name, 
                   config={"epochs": num_epochs, 
                           'train_params': train_params, 
                           'total_params': total_params,
                           'lr': optimizer.param_groups[0]['lr']})
        wandb.watch(self.model, log="all")

        self.checkpoint_path = f"best_model_{experiment_name}.pth"

    def __train(self, epoch):
        """
        Train the model for one epoch
        """
        self.model.train()
        running_loss, running_correct, running_amount = 0.0, 0, 0

        for batch_idx, (inputs, labels) in tqdm(enumerate(self.train_loader), desc=f"Training Epoch {epoch+1}", total=len(self.train_loader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_amount += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

            if batch_idx % self.log_interval == 0:
                train_acc = (running_correct.float() / running_amount).item()
                wandb.log({"Train/Loss": running_loss / running_amount, "Train/Accuracy": train_acc})

            if self.max_iter and batch_idx > self.max_iter:
                break

        train_loss = running_loss / len(self.train_loader.dataset)
        train_acc = running_correct.float() / len(self.train_loader.dataset)
        return train_loss, train_acc

    def __validate(self, epoch):
        """
        Evaluate the model on validation dataset
        """
        self.model.eval()
        running_loss, running_correct = 0.0, 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in tqdm(enumerate(self.val_loader), desc=f"Validating Epoch {epoch+1}", total=len(self.val_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

                if batch_idx % self.log_interval == 0:
                    val_acc = (running_correct.float() / ((batch_idx + 1) * self.val_loader.batch_size)).item()
                    wandb.log({"Validation/Loss": running_loss / ((batch_idx + 1) * self.val_loader.batch_size), "Validation/Accuracy": val_acc})

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = running_correct.float() / len(self.val_loader.dataset)

        return val_loss, val_acc

    def train(self):
        """
        Run the full training loop
        """
        for epoch in range(self.num_epochs):

            train_loss, train_acc = self.__train(epoch)
            val_loss, val_acc = self.__validate(epoch)


            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = deepcopy(self.model.state_dict())

            wandb.log(
                {
                    "Epoch/Train Loss": train_loss,
                    "Epoch/Train Accuracy": train_acc,
                    "Epoch/Validation Loss": val_loss,
                    "Epoch/Validation Accuracy": val_acc,
                }
            )

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        if self.best_model_state:
            torch.save(self.best_model_state, self.checkpoint_path)
            print(f"Best model saved as {self.checkpoint_path}")

        if self.hf_repo_name:
            self.save_to_hf()
        
        wandb.finish()
        return self.best_model_state

    def save_to_hf(self):
        """
        Uploads the best model to Hugging Face Hub with the experiment name
        """
        if self.best_model_state is None:
            print("No best model found. Skipping HF upload")
            return

        api = HfApi(token=HF_TOKEN)
        upload_file(
            repo_id=self.hf_repo_name,
            path_or_fileobj=self.checkpoint_path,
            path_in_repo=f"{self.experiment_name}.pth",
            token=HF_TOKEN,
        )

        print(f"Model uploaded to HF Hub as {self.experiment_name}.pth")


    def count_parameters(self):
        """
        Count number of trainable parameters in the model
        """
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, total_params

