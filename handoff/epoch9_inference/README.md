# Epoch-9 AF to H&E inference

Stage-1 checkpoint `checkpoints/epoch_0009.pt` (epoch 9, step 601068). The generator is in `models/`. Collage construction is in `data/collage.py`. `infer.py` builds the collage and runs the generator.

The input is a 16-bit AF TIFF. Values are divided by 65535 and mapped to [-1, 1]. There is no contrast stretch. The collage is version 0 of the 1024 rule: one thumbnail mosaic, three 2048 fields, four 1024 fields, and eight 256 windows. Tiles are 512 and do not overlap. The output is in [-1, 1] from a tanh head.

```bash
python infer.py --af /path/to/slide-AF-R1.tif --out /path/to/out
```

A directory of TIFFs is accepted as `--af`. Each slide is written to `out/<slide>/collage.png`, `fake_he.jpg`, and `views.png`.

Needs one CUDA GPU, plus the packages in `requirements.txt`. PyTorch has to be a CUDA build.
