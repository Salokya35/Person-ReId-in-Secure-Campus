def extract_features(model, dataloader):
    model.eval()
    features = []
    labels_list = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Store features and labels
            features.append(outputs.cpu())
            labels_list.append(labels.cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels
