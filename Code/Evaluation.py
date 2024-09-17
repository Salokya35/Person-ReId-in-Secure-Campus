def calc_rank_k_accuracy(k, model, query_dir, gallery_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query_dataset = Organize_Dataset(query_dir, transform=transform)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=4)

    gallery_dataset = Organize_Dataset(gallery_dir, transform=transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4)

    query_features, query_labels = extract_features(model, query_loader)
    gallery_features, gallery_labels = extract_features(model, gallery_loader)

    dist_matrix = pairwise_distances(query_features, gallery_features,metric='cosine')

    rank_accuracies = {i: 0 for i in range(1, k+1)}

    for i in range(len(query_labels)):
        sorted_idx = np.argsort(dist_matrix[i])
        for rank in range(1, k+1):
            if query_labels[i] in gallery_labels[sorted_idx[:rank]]:
                rank_accuracies[rank] += 1

    accuracy_list = []
    for rank in range(1, k+1):
        accuracy = rank_accuracies[rank] / len(query_labels)
        accuracy_list.append(accuracy)
        print(f'Rank-{rank} Accuracy: {accuracy:.4f}')
    return accuracy_list

