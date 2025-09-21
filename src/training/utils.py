def prepare_model(model):
    # Function to prepare the model for training
    model.train()
    return model

def log_training_info(epoch, loss):
    # Function to log training information
    print(f"Epoch: {epoch}, Loss: {loss}")

def save_model(model, path):
    # Function to save the trained model
    model.save_pretrained(path)

def load_model(path):
    # Function to load a pre-trained model
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path)