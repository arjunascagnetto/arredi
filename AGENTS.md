# AGENTS.md

## 🎯 Scopo del Sistema

Il sistema ha l'obiettivo di analizzare un'immagine di interni, riconoscere e ritagliare gli oggetti principali (divano, tavolo, tende, ecc.), e per ciascun oggetto cercare online 10 prodotti simili in base alla categoria rilevata. Tutto il flusso è realizzato con tecnologie open-source e può funzionare anche senza GPU.

---

## 🧠 Agenti e Componenti

### 1. **ImageProcessorAgent**
**Responsabilità:**  
- Carica un'immagine fornita (via API o percorso locale)  
- Utilizza YOLOv8 con segmentazione per identificare gli oggetti nell'immagine  
- Ritaglia ogni oggetto usando maschera o bounding box  
- Salva ogni ritaglio in una directory definita

**Input:**  
- Immagine `.jpg` o `.png`  
- Configurazione del modello YOLOv8 (categorie, soglia di confidenza)

**Output:**  
- JSON con categoria, path immagine ritagliata

**Dipendenze:**  
- `ultralytics`, `opencv-python`, `PIL`, `numpy`

---

### 2. **EmbeddingAgent**
**Responsabilità:**  
- Carica le immagini ritagliate  
- Estrae embedding visivi usando il modello `CLIP`  
- Prepara i vettori per la fase di matching

**Input:**  
- Percorsi immagine ritagliata

**Output:**  
- Dizionario `{categoria: embedding_tensor}`

**Dipendenze:**  
- `openai-clip`, `torch`, `transformers`, `PIL`

---

### 3. **ProductSearchAgent**
**Responsabilità:**  
- Riceve l’embedding dell’oggetto target  
- Confronta l’embedding con un database locale di prodotti preindicizzati  
- Restituisce le 10 corrispondenze più simili per categoria

**Input:**  
- Embedding + categoria

**Output:**  
- Lista di 10 prodotti con: titolo, URL, immagine

**Dipendenze:**  
- `faiss-cpu`, `sqlite3`, `pandas`, `json`

---

### 4. **WebScraperAgent** (facoltativo, run periodico)
**Responsabilità:**  
- Visita siti e-commerce predefiniti  
- Raccoglie titoli, immagini, link, e categorie dei prodotti  
- Salva i dati in un database relazionale o formato JSON

**Input:**  
- URL di origine

**Output:**  
- Dataset strutturato per embedding e indicizzazione

**Dipendenze:**  
- `beautifulsoup4`, `requests`, `selenium` (opzionale)

---

## 🔄 Flusso Principale

1. **Input** immagine → `ImageProcessorAgent`  
2. **Output** oggetti ritagliati → `EmbeddingAgent`  
3. **Output** embedding → `ProductSearchAgent`  
4. **Risultato** finale: lista JSON per ciascun elemento trovato

---

## 📁 Output JSON Esempio

```json
[
  {
    "categoria": "диван",
    "immagine": "outputs/elements/divan_001.jpg",
    "prodotti": [
      {
        "titolo": "Современный диван IKEA",
        "url": "https://ikea.ru/product/123",
        "img": "https://ikea.ru/product/123.jpg"
      },
      ...
    ]
  },
  ...
]
