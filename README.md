# interactor-cyclegan-style-transfer

An HTTP model server that restyles a body render as a painting with an upstream unpaired image-to-image GAN, leaving the pose in place.

## What it is for

It makes the stylized branch of the training corpus from renders, using a GAN rather than a diffusion model so the stylized branch does not share the photoreal branch's texture errors. Each result carries the hash of the checkpoint that made it, because the output is generated synthetic data and its provenance travels with the data. It follows the packaging convention of [RFD 1036](https://github.com/V-Sekai-fire/manuals-weftspun/tree/main/rfd/1036-packaging-convention).

## Build and run

```sh
python server.py
```

It needs the upstream generator code on the import path and its pretrained style checkpoints; without a checkpoint a request fails rather than returning an unstyled image.

## Licence

The server is licensed under either of Apache-2.0 or MIT at your option; see `LICENSE-APACHE` and `LICENSE-MIT`. The upstream generator and its pretrained checkpoints keep their own BSD terms.
