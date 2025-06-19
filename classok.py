import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import top_k_accuracy_score
import seaborn as sns

try:
    import umap
except ImportError:
    umap = None

class ImageCaptioner:
    def __init__(self, 
                 model, 
                 processor, 
                 base_folder, 
                 label_map, 
                 output_json_path,
                 prompt,
                 device="cuda",
                 dtype=None):
        self.model = model
        self.processor = processor
        self.base_folder = base_folder
        self.label_map = label_map
        self.output_json_path = output_json_path
        self.prompt_template = prompt
        self.device = device
        self.dtype = dtype
        self.results = []

        if os.path.exists(self.output_json_path):
            with open(self.output_json_path, "r") as f:
                self.results = json.load(f)

    def caption_images_in_folder(self, folder_range=(1, 1)):
        for folder in sorted(os.listdir(self.base_folder)):
            if not folder.isdigit():
                continue

            folder_num = int(folder)
            if not (folder_range[0] <= folder_num <= folder_range[1]):
                continue

            if folder not in self.label_map:
                print(f"Skipping folder {folder}: no label in JSON")
                continue

            label = self.label_map[folder]
            folder_path = os.path.join(self.base_folder, folder)

            if not os.path.isdir(folder_path):
                continue

            for image_name in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing folder {folder}"):
                image_path = os.path.join(folder_path, image_name)
                
                if any(r["image_path"] == image_path for r in self.results):
                    continue  # Already processed

                self._process_image(image_path, folder, label)

    def _process_image(self, image_path, folder, label, max_new_tokens=500):
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = self.prompt_template.format(label=label)

            inputs = self.processor(
                text=prompt.strip(),
                images=[image],
                return_tensors="pt"
            ).to(self.device, dtype=self.dtype)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            self.results.append({
                "image_path": image_path,
                "folder": folder,
                "label": label,
                "caption": caption
            })

            self.save_results()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    def save_results(self):
        with open(self.output_json_path, "w") as f:
            json.dump(self.results, f, indent=2)


class EmbeddingProcessor:
    def __init__(self, model_name, cat_to_name_path, input_json, output_json, dataset_root):
        self.model_name = model_name
        self.cat_to_name_path = cat_to_name_path
        self.input_json = input_json
        self.output_json = output_json
        self.dataset_root = dataset_root

        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name)
        self.model.eval().cuda()  # or .cpu()

        with open(cat_to_name_path, 'r') as f:
            self.cat_to_name = json.load(f)

    def clean_caption(self, caption):
        match = re.search(r'Assistant:.*', caption, re.DOTALL)
        return match.group(0).strip() if match else caption

    def preprocess_json(self, save_path="cleaned_file_updated.json"):
        # Load and clean
        with open(self.input_json, 'r') as f:
            data = json.load(f)

        for item in data:
            item['caption'] = self.clean_caption(item['caption'])

        for item in data:
            parts = item["image_path"].split('/')
            folder = parts[-2]
            filename = parts[-1]
            item["image_path"] = f'{self.dataset_root}/{folder}/{filename}'
            item["folder"] = folder

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Preprocessed and saved to {save_path}")
        return save_path

    def generate_embeddings(self, cleaned_json_path):
        with open(cleaned_json_path, 'r') as f:
            caption_data = json.load(f)

        output_data = []

        for entry in tqdm(caption_data):
            image_path = entry["image_path"]
            caption = entry["caption"]
            folder = entry["folder"]
            label = self.cat_to_name.get(folder, "Unknown")

            # Encode text
            inputs = self.processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs).squeeze().cpu().tolist()

            # Encode image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                image_emb = self.model.get_image_features(**inputs).squeeze().cpu().tolist()

            output_data.append({
                "image path": image_path,
                "label": label,
                "text embedding": text_emb,
                "image embedding": image_emb
            })

        with open(self.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Embeddings saved to {self.output_json}")


class EmbeddingPlotter:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_data()
        self.labels = [entry["label"] for entry in self.data]
        self.text_embeddings = normalize(np.array([entry["text embedding"] for entry in self.data]))
        self.image_embeddings = normalize(np.array([entry["image embedding"] for entry in self.data]))

    def _load_data(self):
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def reduce_embeddings(self, method="pca", use="combined"):
        if use == "text":
            embeddings = self.text_embeddings
        elif use == "image":
            embeddings = self.image_embeddings
        elif use == "combined":
            embeddings = np.concatenate([self.image_embeddings, self.text_embeddings], axis=0)
        else:
            raise ValueError("use must be 'text', 'image', or 'combined'.")

        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        elif method == "umap":
            if umap is None:
                raise ImportError("UMAP is not installed. Run `pip install umap-learn`.")
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("method must be 'pca', 'tsne', or 'umap'.")

        reduced = reducer.fit_transform(embeddings)
        return reduced

    def plot(self, method="pca", use="combined", color_by_label=True):
        reduced = self.reduce_embeddings(method=method, use=use)

        plt.figure(figsize=(12, 10))
        if use == "combined":
            length = len(self.image_embeddings)
            plt.scatter(reduced[:length, 0], reduced[:length, 1], label="Images", alpha=0.5)
            plt.scatter(reduced[length:, 0], reduced[length:, 1], label="Captions", alpha=0.5)
            plt.legend()
        elif color_by_label:
            unique_labels = sorted(set(self.labels))
            palette = sns.color_palette("hsv", len(unique_labels))
            label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

            for label in unique_labels:
                idx = [i for i, l in enumerate(self.labels) if l == label]
                plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, color=label_to_color[label], s=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)

        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)

        plt.title(f"{method.upper()} of {use.capitalize()} Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class EmbeddingClassifier:
    def __init__(self, json_path, embedding_type='text embedding', model_type='logistic', test_size=0.1):
        """
        Initialize the classifier.
        """
        self.json_path = json_path
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.test_size = test_size
        self.model = self._get_model()
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def _get_model(self):
        """
        Initialize the model.
        """
        if self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier()
        elif self.model_type == 'svm':
            return SVC()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def load_data(self):
        """
        Load data from the JSON file.
        """
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        embeddings = []
        labels = []

        for item in data:
            embeddings.append(item[self.embedding_type])
            labels.append(item['label'])

        self.X = np.array(embeddings)
        self.y = np.array(labels)

    def split_data(self):
        """
        Split the data using stratified sampling.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=42
        )

    def train(self):
        """
        Train the model.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """
        y_pred = self.model.predict(self.X_test)
        print(f"\n--- Evaluation Report for {self.embedding_type} using {self.model_type} ---")
        print(classification_report(self.y_test, y_pred))

    def run(self):
        """
        Run the full pipeline: load, split, train, evaluate.
        """
        self.load_data()
        self.split_data()
        self.train()
        self.evaluate()


class EmbeddingSimilarity:
    def __init__(self, json_path):
        self.json_path = json_path
        self.text_embeddings = []
        self.image_embeddings = []
        self.labels = []
        self.image_paths = []

    def load_embeddings(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        for item in data:
            self.text_embeddings.append(item["text embedding"])
            self.image_embeddings.append(item["image embedding"])
            self.labels.append(item["label"])
            self.image_paths.append(item["image path"])

        self.text_embeddings = np.array(self.text_embeddings)
        self.image_embeddings = np.array(self.image_embeddings)
        self.labels = np.array(self.labels)

    def compute_similarity_and_recall(self, k=5):
        """
        Compute cosine similarity between text and image embeddings.
        Return Top-1 and Top-k accuracy.
        """
        sim_matrix = cosine_similarity(self.text_embeddings, self.image_embeddings)

        # Get sorted indices of most similar images for each text
        top_k_preds = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k]

        # Ground truth: correct image is at the same index
        gt_indices = np.arange(len(self.labels))  # assuming 1-to-1 mapping

        top1_correct = np.any(top_k_preds[:, :1] == gt_indices[:, None], axis=1)
        topk_correct = np.any(top_k_preds == gt_indices[:, None], axis=1)

        top1_acc = np.mean(top1_correct)
        topk_acc = np.mean(topk_correct)

        print(f"\nTop-1 Accuracy (Recall@1): {top1_acc:.4f}")
        print(f"Top-{k} Accuracy (Recall@{k}): {topk_acc:.4f}")

        return top_k_preds, gt_indices

    def show_examples(self, top_k_preds, num_examples=5):
        """
        Plot text, real image, and Top-1 retrieved image.
        """
        for idx in range(num_examples):
            true_idx = idx
            pred_idx = top_k_preds[idx][0]

            true_img = Image.open(self.image_paths[true_idx])
            pred_img = Image.open(self.image_paths[pred_idx])
            label = self.labels[true_idx]

            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            axs[0].imshow(true_img)
            axs[0].set_title("True Image")
            axs[0].axis("off")

            axs[1].imshow(pred_img)
            axs[1].set_title("Top-1 Match")
            axs[1].axis("off")

            plt.suptitle(f"Label: {label}", fontsize=12)
            plt.show()

    def run_retrieval_evaluation(self, k=5, num_examples=5):
        self.load_embeddings()
        top_k_preds, gt = self.compute_similarity_and_recall(k=k)
        self.show_examples(top_k_preds, num_examples=num_examples)

