import os
import json
import re
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from transformers import SiglipProcessor, SiglipModel
from torchvision.transforms import v2 as transforms_v2

try:
    import umap
except ImportError:
    umap = None

class ImageCaptioner:
    """
    A general-purpose class for generating captions from images using vision-language models
    like IDEFICS, BLIP, or LLaVA.
    """

    def __init__(self, 
                 model, 
                 processor, 
                 base_folder, 
                 label_map, 
                 output_json_path,
                 prompt,
                 model_type="idefics",  # NEW: Switch between BLIP, LLaVA, etc.
                 device="cuda",
                 dtype=None):
        """
        Initialize the captioning model.

        Args:
            model: The vision-language model to use (HuggingFace model)
            processor: Corresponding processor/tokenizer
            base_folder: Base path to the image folders
            label_map: Dict mapping folder names to flower labels
            output_json_path: Path to save the caption output JSON
            prompt: Prompt template (supports `{label}` formatting)
            model_type: One of ["idefics", "blip", "llava"]
            device: "cuda" or "cpu"
            dtype: Torch data type (e.g., float16 for GPU acceleration)
        """
        self.model = model
        self.processor = processor
        self.base_folder = base_folder
        self.label_map = label_map
        self.output_json_path = output_json_path
        self.prompt_template = prompt
        self.model_type = model_type.lower()
        self.device = device
        self.dtype = dtype
        self.results = []

        # Load previous results if available
        if os.path.exists(self.output_json_path):
            with open(self.output_json_path, "r") as f:
                self.results = json.load(f)

    def caption_single_image(self, image_path, max_new_tokens=500):
        folder = os.path.basename(os.path.dirname(image_path))
        if folder not in self.label_map:
            raise ValueError(f"Label not found for folder: {folder}")
        label = self.label_map[folder]
        prompt = self.prompt_template.format(label=label)
    
        try:
            image = Image.open(image_path).convert("RGB")
    
            # === LLaVA ===
            if self.model_type == "llava":
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
    
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3
                )
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
            # === IDEFICS ===
            elif self.model_type == "idefics":
                inputs = self.processor(
                    text=prompt,
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
    
            # === BLIP2 ===
            elif self.model_type == "blip2":
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
    
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3
                )
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
    
            self.results.append({
                "image_path": image_path,
                "folder": folder,
                "label": label,
                "caption": caption
            })
            self.save_results()
            return caption
    
        except Exception as e:
            print(f"Error while processing {image_path}: {e}")
            return None




    def caption_selected_images(self, image_paths):
        """
        Caption only the specific images passed in as paths.

        Args:
            image_paths: List of full image paths (strings)
        """
        for image_path in image_paths:
            print(f"\nCaptioning: {image_path}")
            self.caption_single_image(image_path)

    def caption_images_in_folder(self, folder_range=(1, 1)):
        """
        Caption all images within a folder or a range of numbered folders.
        """
        for folder in sorted(os.listdir(self.base_folder)):
            if not folder.isdigit():
                continue
            folder_num = int(folder)
            if not (folder_range[0] <= folder_num <= folder_range[1]):
                continue
            if folder not in self.label_map:
                print(f"Skipping {folder}: no label in label_map")
                continue
            folder_path = os.path.join(self.base_folder, folder)
            for image_name in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing folder {folder}"):
                image_path = os.path.join(folder_path, image_name)
                if any(r["image_path"] == image_path for r in self.results):
                    continue
                self.caption_single_image(image_path)

    def save_results(self):
        """
        Save all captioning results to the output JSON file.
        """
        with open(self.output_json_path, "w") as f:
            json.dump(self.results, f, indent=2)


class EmbeddingProcessor:
    def __init__(self, model_name, cat_to_name_path, input_json, output_json, dataset_root):
        self.model_name = model_name
        self.cat_to_name_path = cat_to_name_path
        self.input_json = input_json
        self.output_json = output_json
        self.dataset_root = dataset_root

        # Load processor and model
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name)
        self.model.eval().cuda()

        # Load label map
        with open(cat_to_name_path, 'r') as f:
            self.cat_to_name = json.load(f)

        # Define torchvision-based random resized crop
        self.image_transform = transforms_v2.Compose([
            transforms_v2.Resize(256, antialias=True),
            transforms_v2.RandomResizedCrop(size=(224, 224), antialias=True),
        ])

    def clean_caption(self, caption):
        """Extract the useful part from the caption (usually after 'Assistant:')."""
        match = re.search(r'Assistant:.*', caption, re.DOTALL)
        return match.group(0).strip() if match else caption

    def preprocess_json(self, save_path="cleaned_file_updated.json"):
        """Fix paths and clean captions in the input JSON."""
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

    def generate_embeddings(self, cleaned_json_path, apply_augmentation=True):
        """Generate and save text and image embeddings."""
        with open(cleaned_json_path, 'r') as f:
            caption_data = json.load(f)

        output_data = []

        for entry in tqdm(caption_data):
            image_path = entry["image_path"]
            caption = entry["caption"]
            folder = entry["folder"]
            label = self.cat_to_name.get(folder, "Unknown")

            # Text embedding
            inputs = self.processor(
                text=caption,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs).squeeze().cpu().tolist()

            # Image embedding with optional augmentation
            image = Image.open(image_path).convert("RGB")
            if apply_augmentation:
                image = self.image_transform(image)

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
    """Beágyazások vizualizációjára szolgáló osztály."""
    
    def __init__(self, json_path):
        """
        Inicializálás.
        
        Args:
            json_path: A beágyazásokat tartalmazó JSON fájl elérési útja
        """
        self.json_path = json_path
        self.data = self._load_data()
        self.labels = [entry["label"] for entry in self.data]
        self.text_embeddings = normalize(np.array([entry["text embedding"] for entry in self.data]))
        self.image_embeddings = normalize(np.array([entry["image embedding"] for entry in self.data]))

    def _load_data(self):
        """Adatok betöltése JSON fájlból."""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def reduce_embeddings(self, method="pca", use="combined"):
        """
        Beágyazások dimenzionalitásának csökkentése.
        
        Args:
            method: Csökkentés módszere ('pca', 'tsne' vagy 'umap')
            use: Milyen beágyazásokat használjon ('text', 'image' vagy 'combined')
            
        Returns:
            A csökkentett dimenziójú beágyazások
        """
        if use == "text":
            embeddings = self.text_embeddings
        elif use == "image":
            embeddings = self.image_embeddings
        elif use == "combined":
            embeddings = np.concatenate([self.image_embeddings, self.text_embeddings], axis=0)
        else:
            raise ValueError("A 'use' paraméternek 'text', 'image' vagy 'combined' kell legyen.")

        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        elif method == "umap":
            if umap is None:
                raise ImportError("Az UMAP nincs telepítve. Telepítsd a 'pip install umap-learn' paranccsal.")
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("A 'method' paraméternek 'pca', 'tsne' vagy 'umap' kell legyen.")

        reduced = reducer.fit_transform(embeddings)
        return reduced

    def plot(self, method="pca", use="combined", color_by_label=True):
        """Beágyazások ábrázolása."""
        reduced = self.reduce_embeddings(method=method, use=use)

        plt.figure(figsize=(12, 10))
        if use == "combined":
            length = len(self.image_embeddings)
            plt.scatter(reduced[:length, 0], reduced[:length, 1], label="Képek", alpha=0.5)
            plt.scatter(reduced[length:, 0], reduced[length:, 1], label="Feliratok", alpha=0.5)
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

        plt.title(f"{method.upper()} {use.capitalize()} beágyazások")
        plt.xlabel("Komponens 1")
        plt.ylabel("Komponens 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class EmbeddingClassifier:
    """Beágyazások osztályozására szolgáló osztály."""
    
    def __init__(self, json_path, embedding_type='text embedding', model_type='logistic', test_size=0.1):
        """
        Inicializálás.
        
        Args:
            json_path: JSON fájl elérési útja
            embedding_type: Használandó beágyazás típusa ('text embedding' vagy 'image embedding')
            model_type: Modell típusa ('logistic', 'random_forest' vagy 'svm')
            test_size: Teszt adathalmaz mérete
        """
        self.json_path = json_path
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.test_size = test_size
        self.model = self._get_model()
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def _get_model(self):
        """Modell inicializálása a megadott típus alapján."""
        if self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier()
        elif self.model_type == 'svm':
            return SVC()
        else:
            raise ValueError(f"Ismeretlen modell típus: {self.model_type}")

    def load_data(self):
        """Adatok betöltése a JSON fájlból."""
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
        """Adatok felosztása tanító és teszt halmazra."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=42
        )

    def train(self):
        """Modell tanítása."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Modell kiértékelése."""
        y_pred = self.model.predict(self.X_test)
        print(f"\n--- Kiértékelési jelentés {self.embedding_type} használatával {self.model_type} modell esetén ---")
        print(classification_report(self.y_test, y_pred))

    def run(self):
        """Teljes folyamat futtatása: betöltés, felosztás, tanítás, kiértékelés."""
        self.load_data()
        self.split_data()
        self.train()
        self.evaluate()


class EmbeddingSimilarityEvaluator:
    """
    Egyesített osztály képszintű és osztályszintű hasonlóságvizsgálathoz,
    támogatja a koszinusz és CSLS metrikákat, valamint példák megjelenítését is.
    """

    def __init__(self, json_path):
        self.json_path = json_path
        self.text_embeddings = []
        self.image_embeddings = []
        self.labels = []
        self.image_paths = []
        self.results = {}

    def load_embeddings(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        for item in data:
            self.text_embeddings.append(item["text embedding"])
            self.image_embeddings.append(item["image embedding"])
            self.labels.append(item["label"])
            self.image_paths.append(item.get("image path", None))  # Optional

        self.text_embeddings = np.array(self.text_embeddings)
        self.image_embeddings = np.array(self.image_embeddings)
        self.labels = np.array(self.labels)

    def _csls_matrix(self, csls_k=10):
        sim_matrix = cosine_similarity(self.text_embeddings, self.image_embeddings)
        text_to_img = cosine_similarity(self.text_embeddings, self.image_embeddings)
        img_to_text = cosine_similarity(self.image_embeddings, self.text_embeddings)

        r_text = np.mean(-np.sort(-text_to_img, axis=1)[:, :csls_k], axis=1)[:, np.newaxis]
        r_image = np.mean(-np.sort(-img_to_text, axis=1)[:, :csls_k], axis=1)[np.newaxis, :]

        csls_matrix = 2 * sim_matrix - r_text - r_image
        return csls_matrix

    def _evaluate_image_retrieval(self, sim_matrix, k=5):
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k]
        top1_correct = top_k_indices[:, 0] == np.arange(len(self.labels))
        topk_correct = np.any(top_k_indices == np.arange(len(self.labels))[:, None], axis=1)
        return top_k_indices, np.mean(top1_correct), np.mean(topk_correct)

    def _evaluate_class_retrieval(self, sim_matrix, k=5):
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k]
        top_k_labels = self.labels[top_k_indices]
        correct_labels = self.labels[:, None]
        top1_correct = np.any(top_k_labels[:, :1] == correct_labels, axis=1)
        topk_correct = np.any(top_k_labels == correct_labels, axis=1)
        return top_k_labels, np.mean(top1_correct), np.mean(topk_correct)

    def evaluate_all(self, k=5):
        cos_sim = cosine_similarity(self.text_embeddings, self.image_embeddings)
        csls_sim = self._csls_matrix()

        _, top1_cos_img, top5_cos_img = self._evaluate_image_retrieval(cos_sim, k)
        _, top1_csls_img, top5_csls_img = self._evaluate_image_retrieval(csls_sim, k)

        _, top1_cos_cls, top5_cos_cls = self._evaluate_class_retrieval(cos_sim, k)
        _, top1_csls_cls, top5_csls_cls = self._evaluate_class_retrieval(csls_sim, k)

        self.results = {
            "Captionings": os.path.basename(self.json_path),
            "top1cosine-image": top1_cos_img,
            "top5cosine-image": top5_cos_img,
            "top1csls-image": top1_csls_img,
            "top5csls-image": top5_csls_img,
            "top1cosine-class": top1_cos_cls,
            "top5cosine-class": top5_cos_cls,
            "top1csls-class": top1_csls_cls,
            "top5csls-class": top5_csls_cls
        }

        return self.results

    def show_examples(self, sim_matrix, k=5, num_examples=5,
                  labels_to_show=None, show_correct=False,
                  match_on_class=False):  # <- new parameter

      top_k_preds = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k]
      shown = 0

      for idx in range(len(self.labels)):
          true_idx = idx
          pred_idx = top_k_preds[idx][0]

          if labels_to_show and self.labels[true_idx] not in labels_to_show:
              continue

          # Choose match method
          if match_on_class:
              is_correct = self.labels[true_idx] == self.labels[pred_idx]
          else:
              is_correct = true_idx == pred_idx

          if not show_correct and is_correct:
              continue
          if show_correct and not is_correct:
              continue

          if not self.image_paths[true_idx] or not self.image_paths[pred_idx]:
              continue

          true_img = Image.open(self.image_paths[true_idx])
          pred_img = Image.open(self.image_paths[pred_idx])

          fig, axs = plt.subplots(1, 2, figsize=(6, 3))
          axs[0].imshow(true_img)
          axs[0].set_title("Valódi kép")
          axs[0].axis("off")

          axs[1].imshow(pred_img)
          axs[1].set_title("Top-1 találat")
          axs[1].axis("off")

          plt.suptitle(f"Címke: {self.labels[true_idx]}", fontsize=12)
          plt.show()

          shown += 1
          if shown >= num_examples:
              break

    def run_retrieval_evaluation(self, k=5, num_examples=5, labels_to_show=None, show_correct=False, use_csls=False):
        self.load_embeddings()
        sim_matrix = self._csls_matrix() if use_csls else cosine_similarity(self.text_embeddings, self.image_embeddings)
        self.show_examples(sim_matrix, k=k, num_examples=num_examples, labels_to_show=labels_to_show, show_correct=show_correct)


