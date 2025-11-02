# DCASE SED Codebase - Quick Reference Guide

## Critical File Paths (Absolute)

### Model Architecture
- **CRNN Model:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/CRNN.py`
- **CNN Backbone:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/CNN.py`
- **RNN Modules:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/RNN.py`
- **BEATs Embedder:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/beats/BEATs.py`
- **MixStyle Modules:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/mixstyle.py`

### Training & Inference
- **Main Trainer:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/sed_trainer_pretrained.py`
- **Training Script:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/train_pretrained.py`
- **Embedding Extraction:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/extract_embeddings.py`

### Data Processing
- **Dataset Classes:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/dataio/datasets.py`
- **Alternative Datasets:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/dataio/pre_datasets.py`
- **Data Augmentation:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/data_augm.py`
- **Label Encoder:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/utils/encoder.py`
- **Scaler (Normalization):** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/utils/scaler.py`

### Post-processing
- **Median Filter:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/utils/postprocess.py`
- **cSEBBs:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/sebbs/sebbs/csebbs.py`

### Configuration
- **Main Config:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/confs/pretrained.yaml`
- **Alternative Config:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/confs/before_pretrained.yaml`

---

## Code Snippets

### 1. Load Pre-trained Model for Inference

```python
import torch
from local.sed_trainer_pretrained import SEDTask4
import yaml

# Load configuration
with open('confs/pretrained.yaml') as f:
    config = yaml.safe_load(f)

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.ckpt')
model = SEDTask4.load_from_checkpoint('path/to/checkpoint.ckpt', hparams=config)

# Set to evaluation mode
model.eval()
```

### 2. Perform Inference on Audio File

```python
import torchaudio
import torch

# Load audio
audio_path = 'path/to/audio.wav'
audio, sr = torchaudio.load(audio_path)

# Resample to 16kHz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)

# Extract mel-spectrogram
mels = model.mel_spec(audio.unsqueeze(0))  # Add batch dimension

# Load embeddings from HDF5 if using them
import h5py
embeddings_file = 'embeddings/beats/filename.hdf5'
with h5py.File(embeddings_file, 'r') as f:
    frame_emb = torch.from_numpy(f['frame_embeddings'][...]).unsqueeze(0)

# Forward pass
with torch.no_grad():
    strong_pred, weak_pred = model.detect(mels, model.sed_student, embeddings=frame_emb)

# Post-processing
scores = model.median_filter(strong_pred[0].cpu().numpy().T)  # (frames, classes)
```

### 3. Extract BEATs Embeddings

```python
from extract_embeddings import extract, WavDataset
import torch

# Create dataset
dataset = WavDataset(folder='path/to/audio/', pad_to=10, fs=16000)

# Extract embeddings
embedding_model = load_beats_model()  # Your BEATs model
extract(
    batch_size=32,
    folder='path/to/audio/',
    dset_name='my_dataset',
    torch_dset=dataset,
    embedding_model=embedding_model,
    use_gpu=True
)
```

### 4. Create Custom Model

```python
from desed_task.nnet.CRNN import CRNN

model = CRNN(
    n_in_channel=1,
    nclass=27,
    attention=True,
    activation="glu",
    dropout=0.2,
    rnn_type="BGRU",
    n_RNN_cell=192,
    use_embeddings=True,
    embedding_size=768,
    embedding_type='frame',
    aggregation_type='pool1d',
    kernel_size=[3, 3, 3, 3, 3, 3, 3],
    nb_filters=[16, 32, 64, 128, 128, 128, 128],
    pooling=[[2,2], [2,2], [1,2], [1,2], [1,2], [1,2], [1,2]],
    n_layers_RNN=1,
    specaugm_t_p=0.0,
    specaugm_f_p=0.0,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 5. Train Custom Model

```bash
# Using uv (project uses uv for dependency management)
uv run train_pretrained.py \
    --mixstyle_type moreMix \
    --wandb_dir my_experiment \
    --cmt  # Enable Confident Mean Teacher \
    --sebbs  # Enable cSEBBs post-processing
```

### 6. Data Loading

```python
from desed_task.dataio.datasets import StronglyAnnotatedSet, WeakSet
from desed_task.utils.encoder import ManyHotEncoder

# Create encoder
encoder = ManyHotEncoder(
    labels=['Alarm', 'Speech', 'Music', ...],
    audio_len=10,
    frame_len=2048,
    frame_hop=256,
    net_pooling=4,
    fs=16000,
)

# Load dataset
strong_dataset = StronglyAnnotatedSet(
    audio_folder='path/to/audio/',
    tsv_entries=df,
    encoder=encoder,
    pad_to=10,
    fs=16000,
    feats_pipeline=None,
    embeddings_hdf5_file='embeddings/beats/strong.hdf5',
    embedding_type='frame',
)
```

---

## Key Parameters Summary

### Network Architecture
```
Input: (batch, 1, 160000)  # Raw audio, 10 seconds at 16kHz

Mel-spectrogram: (batch, 128, 625)  # 128 mel bins, ~625 frames

CNN Layers: 7 convolutional blocks
  - Filters: [16, 32, 64, 128, 128, 128, 128]
  - Kernel: All 3x3
  - Pooling: [[2,2], [2,2], [1,2], [1,2], [1,2], [1,2], [1,2]]
  - After 7 layers: (batch, 128, ~39 frames)  # ~4x downsampling

RNN: Bidirectional GRU
  - Input: 128 (from CNN)
  - Hidden: 192
  - Output: 384 (2 * 192 bidirectional)

Output Heads:
  - Strong (frame-level): (batch, 384) → (batch, 27)
  - Weak (clip-level): (batch, 384) → (batch, 27)
```

### Training Parameters
- Learning rate: 0.001
- Optimizer: Adam
- Batch size: Total 60 (mixed datasets)
- Epochs: 300
- Validation interval: 10 epochs
- Early stopping: 200 epochs patience
- EMA factor: 0.999

### Feature Parameters
- Mel bins: 128
- FFT size: 2048
- Hop length: 256 samples (~16ms)
- Sample rate: 16000 Hz
- Frequency range: 0-8000 Hz

---

## Common Issues & Solutions

### Issue 1: Out of Memory (OOM)
**Solution:** Reduce batch sizes in config
```yaml
training:
  batch_size: [8, 4, 4, 8, 16]  # Reduce from [12, 6, 6, 12, 24]
```

### Issue 2: Missing Embeddings
**Solution:** Extract embeddings first
```bash
python extract_embeddings.py --output_dir ./embeddings
```

### Issue 3: Labels Don't Match Classes
**Solution:** Check encoder configuration matches your dataset
```python
# Ensure encoder classes match your TSV labels
encoder = ManyHotEncoder(list(your_labels_dict.keys()), ...)
```

### Issue 4: Slow Training
**Solution:** 
- Enable GPU: `--gpus 1`
- Reduce validation frequency
- Use mixed precision: `--precision 16`

---

## Evaluation Metrics

### PSDS (Polyphonic Sound Detection Score)
- Best for SED tasks
- Considers temporal and spatial overlap
- Used for early stopping

### F1 Score
- Frame-level for strong labels
- Segment-wise for weak labels
- Computed per-class

### Intersection-based F1
- Alternative metric
- Uses temporal collar tolerance

---

## File Organization Quick View

```
dcase2024_task4_baseline/
├── Models in: desed_task/nnet/
├── Training in: local/sed_trainer_pretrained.py
├── Data in: desed_task/dataio/
├── Config in: confs/
├── Main entry: train_pretrained.py
└── Post-proc: sebbs/sebbs/
```

