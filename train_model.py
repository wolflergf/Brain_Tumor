from src.model_trainer import BrainTumorTrainer
import os

# Ensure the dataset path is correct relative to where the script will be run
dataset_path = os.path.join(".", "brain_tumor_classification", "Training")
trainer = BrainTumorTrainer(dataset_dir=dataset_path)

print("Starting model training...")
trainer.train(epochs=20)
print("Model training complete. Evaluating model...")
trainer.evaluate()

model_save_path = os.path.join(".", "brain_tumor_model.keras")
trainer.save_model(model_save_path)
print(f"Model saved to {model_save_path}")
