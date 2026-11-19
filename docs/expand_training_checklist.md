# Expand training after epoch 7

Checklist for the next training run. Do not start until epoch 7 has saved.

Current run: 302 slides, 237,189 patches. Stop it when `checkpoints/stage1/epoch_0007.pt` is written, before epoch 8 starts. That file is the resume checkpoint and the model that writes the new pseudo-labels.

## Set

- [ ] Add 123 slides, 83,005 patches.
  - [ ] Benign: 54 slides, 39,332 patches
  - [ ] MIA: 59 slides, 34,976 patches
  - [ ] LUSC: 10 slides, 8,697 patches
- [ ] Leave out `F26-02242-1` (malignant, subtype OTHER, no specific type).
- [ ] Leave the exclusion list in force.
- [ ] Do not add the extra LUAD slides.
- [ ] Resulting set: 425 slides, 320,194 patches.
- [ ] Seven of the new slides are batch 1. The other 116 are batch 2.
- [ ] AF patches, real H&E patches, three collages, and crop boxes already exist for all 123.
- [ ] No pseudo-label file exists for any of the 123 yet.

## Roles

- [ ] `epoch_8.pth` writes the teacher images.
- [ ] `epoch_0007.pt` writes the pseudo-label images.
- [ ] The original 302 slides keep the pseudo-labels they already have.
- [ ] Reconstruction stays as it is.
  - [ ] Tissue target: blurred teacher hematoxylin and eosin.
  - [ ] Glass target: blurred pseudo-label hematoxylin and eosin.
  - [ ] Tissue map: AF presence times the stronger of the two tissue scores.
  - [ ] Both sides blurred at sigma 8. Weight 0.2.
- [ ] FDL weight stays 0.003.
- [ ] UNI ramp stays 0 at cosine 0.3 and 1 at cosine 0.6. Weight 0.1.

## 1. Teacher and pseudo-label inference

These two jobs do not depend on each other. Run them at the same time on separate GPUs. Both read the whole-slide TIFF, not the AF patch files.

- [ ] Read each AF slide inside its crop box.
- [ ] Step a 512 window by 384.
- [ ] The center 384 by 384 overwrites.
- [ ] The outer 64 pixels are written only where no center has covered that pixel.
- [ ] Save the assembled 1024 patches on the training grid.
- [ ] Old teacher context: the 1024 field around the crop, plus its WSI token.
- [ ] Epoch-7 context: the same 512 window, plus that slide's existing collage.
- [ ] Teacher files go into the existing teacher cache.
- [ ] Pseudo-label files go into the pseudo-label folder, same filenames as the AF patches.
- [ ] Batch-2 slides use the batch-2 AF thumbnails. Their crop boxes are already in the crop manifest.
- [ ] Do not overwrite teacher files for the original 302 slides.

## 2. UNI table

Start only after both caches exist.

- [ ] Add rows for the 123 slides only.
- [ ] Encode the real H&E and the new teacher image.
- [ ] Take the pseudo-label tissue fraction from the epoch-7 image.
- [ ] Choose the positive with the same local window and the same tissue-gap rule as the current table.
- [ ] Leave the existing 237,189 positives unchanged.

## 3. Negatives

Rebuild the list for every row, old and new. Embeddings for the old rows are already stored.

- [ ] Take up to 256 negatives.
- [ ] First: other slides, same subtype, same tissue bin as the positive.
- [ ] Tissue bin: the chosen positive crop, including a shifted crop. Fraction at or above 0.15 is tissue. Below 0.15 is background.
- [ ] Skip the 8 patches most similar to the positive.
- [ ] If that pool is short: other slides of the same benign or malignant class, same tissue bin.
- [ ] Same slide only for what is still missing. Exclude the positive window and those 8 nearest patches.
- [ ] Benign, LUSC, and MIA can fill the list from other slides.
- [ ] The two SCLC slides, and the one OTHER slide already in training, reach the later steps.

## 4. Training

- [ ] Admit these 123 slides. The loader currently keeps only `crop_absolute` slides.
- [ ] Their AF and real H&E files are already in the training folders.
- [ ] Resume from `epoch_0007.pt`.
- [ ] Run epochs 8 and 9.
- [ ] Keep reconstruction at full weight for both epochs.
- [ ] One epoch will be about a third longer.

## Order

- [ ] 1. Stop at the epoch 7 checkpoint.
- [ ] 2. Teacher inference and pseudo-label inference, together.
- [ ] 3. UNI rows for the 123 slides.
- [ ] 4. Rebuild negatives.
- [ ] 5. Resume training.
