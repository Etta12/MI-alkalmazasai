# MI-alkalmazasai

# Virágok képszintű és feliratszintű hasonlóságelemzése

Ez a projekt egy többkomponensű Python-alapú rendszer, amely virágokról készült képek automatikus leírását (captioning), valamint azok embeddingjeinek vizualizációját, összehasonlítását és klaszterezését valósítja meg.

---

## Projekt moduljai

### 1. Képaláírások generálása (Captioning)
A `ImageCaptioner` osztály segítségével a következő modellekkel készíthetők képaláírások:
- **IDEFICS**
- **BLIP2**
- **LLaVA**

A leírások sablonalapú promptból generálódnak, és mentésre kerülnek JSON fájlba. Támogatott:
- Címke felhasználásával (`label` beillesztve a promptba)
- Címke nélkül (általános prompt)

### 2. Embeddingek generálása képekből és feliratokból
A `EmbeddingProcessor` osztály képes:
- Képek és captionök embeddingjeit kinyerni a SigLIP modellel 
- Eredmények mentése JSON formátumban
- Kép augmentációs lehetőség: szegélyvágás (`edge crop`) — **nem bizonyult hatékonynak**

### 3. Embeddingek vizualizációja
A `EmbeddingPlotter` osztály támogatja:
- **PCA** és **t-SNE** (illetve opcionálisan UMAP) használatát
- Kép és szöveg embeddingek kombinált vagy külön ábrázolását
- Címke szerinti színezést

### 4. Embeddingek klaszterezése és osztályozása
A `EmbeddingClassifier` osztály a következő modellekkel képes klasztereket megtanulni:
- Logistic Regression
- Random Forest
- SVM


### 5. Hasonló képek visszakeresése
A `EmbeddingSimilarityEvaluator` osztály értékeli, hogy egy szöveges embedding:
- **Top-1** és **Top-5** legközelebbi képhez milyen pontossággal talál vissza
- **CSLS** és **koszinusz hasonlóság** metrikákkal dolgozik
- Képes vizualizálni helyes és hibás találatokat
- Támogatja a csak adott címkék szerinti szűrést is

