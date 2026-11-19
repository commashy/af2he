# Fresh WSI coordinates

Batch-1 training patches come from `fresh_patchify.py`. A file

`{date}_{specimen}__y{rel_y}_x{rel_x}.png`

is a 1024 crop whose top-left is **inside the AF crop**, not at the raw WSI origin. The crop box is `af_crop_box` in `crop_manifest_for_thumbs.csv` (`x1,y1,x2,y2`). The absolute box on the AF slide is

```
x1 + rel_x, y1 + rel_y, x1 + rel_x + 1024, y1 + rel_y + 1024
```

The H&E patch uses the same relative offset inside `he_crop_box`. Both boxes are a center crop to the shared width and height.

## Pseudo labels

`images/202610_inference_training_set` holds one JPEG per batch-1 specimen. It sits on the AF canvas. Most files are the AF size, or up to 64 px smaller on the right and bottom. Do not resize a JPEG onto the AF slide.

For a patch, crop the JPEG with the absolute box above. Use the patch only when that box lies inside the JPEG. `scripts/prepare_pseudo_wsi.py` writes `manifests/pseudo_alignment.csv`. Rows with `use=crop_absolute` are safe. Rows with `use=hold` are not: the JPEG is a smaller region (`-AF-R1-1`) or the slide has no crop manifest.

Slides with no JPEG are not all exclusions. Breast and thyroid specimens in `images/fresh_lung/exclusion_ledger.csv` are absent, which matches the training rule. Five malignant lung slides are also absent and are not on that list.

## H&E WSIs

`Fresh_section/HE-WSI` is organized like `AF_batch1`, one file per specimen under a date folder. Compare presence and pixel size with the AF TIFF before cropping. A transferred file can exist before its dimensions match the crop manifest.

## Collages

AF thumbnails for every training slide are in

`images/fresh_lung/batch1/thumbnails/AF` and `batch2/thumbnails/AF`.

They are 1024 overview images, not the patch mosaic.
