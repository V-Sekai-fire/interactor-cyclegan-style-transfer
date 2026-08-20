# interactor-cyclegan-style-transfer

Model image for `cyclegan_style_transfer`, per
[weftspun's RFD 0036](https://github.com/weftspun/request-for-discussion/tree/main/0036-packaging-convention)
packaging convention.

## Model

| Property | Value |
|---|---|
| Upstream | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| License | **BSD** — read from the LICENSE file, not the badge. GitHub reports `NOASSERTION` only because the file concatenates two texts: BSD-2 for CycleGAN, BSD-3 for bundled Facebook DCGAN code. Neither is NC or ND. |
| Parameters | 11.4 M per direction, ResNet-9block generator |
| Ship format | fp32 `.pth`, ~45 MB per style — small enough that quantization buys nothing |

Pretrained styles used here are upstream's own: `style_monet`, `style_ukiyoe`. Both are
one-directional photo→painting generators.

## Why this and not a diffusion restyler

SDPose-OOD built its OOD sets with CycleGAN and StyTR², and said why: *"to avoid introducing
priors from large-scale pretrained diffusion models"*. Their sets are an evaluation benchmark,
so a stylizer carrying a diffusion model's own idea of a person would have measured the prior
rather than the robustness.

The same reasoning holds here for a different reason. The photoreal branch of the corpus is
already diffusion (Qwen-Image-Edit). Driving the stylised branch from the same family would
give every domain one set of texture statistics, hands and faces — a common-mode error rather
than independent ones. CycleGAN is a GAN, trained on different data, by different people. That
independence is the point, and it is why StyTR² being unlicensed cost something real: it was
the second non-diffusion family.

## What it is not

Not a general style transfer. A CycleGAN checkpoint learns **one** unpaired domain mapping and
cannot take a style exemplar at inference — the style is baked into the weights. New target
domains mean new training runs, not new prompts. `AdaAttN` (Apache-2.0), `CAST` (Apache-2.0)
and `EFDM` (MIT) are the exemplar-driven alternatives if that flexibility is wanted. All three
are feed-forward feature transfer, so they remain non-diffusion.

## Interface

`POST /predict`:

| Input | Type | Default | Note |
|---|---|---|---|
| `image` | Path/URL/base64 | required | the ANNY render, or any content image |
| `style` | str | `monet` | `monet` or `ukiyoe` |
| `load_size` | int | 256 | upstream trained at 256. Larger inputs are resized, not tiled |

Returns `{"image": <base64 png>, "style": <str>, "checkpoint_sha256": <str>}`.

The checkpoint hash is returned per result rather than logged once, because the generated
corpus is *generated* synthetic under the workspace's data rule and condition 1 requires the
generating checkpoint recorded **with the data**. A hash in a server log is not with the data.

## Geometry is the thing to check

Stylization is only useful here if it changes appearance and leaves pose alone. That is the
claim SDPose-OOD makes — *"a significant appearance shift while keeping pose-related geometric
information intact"* — and it is a claim, not a guarantee, so it is measured rather than
assumed: `pose-consensus`'s soft silhouette and soft depth compare a stylised frame against the
render it came from, and MediaPipe reads the pose back. A style that moves a limb is a style
that invalidated its own labels.

## Licence

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

`SPDX-License-Identifier: Apache-2.0 OR MIT`

This covers the server and its packaging. The CycleGAN implementation and the pretrained
`style_monet` and `style_ukiyoe` checkpoints are upstream's, under the BSD terms read from
their own LICENSE file and recorded above.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you shall be dual licensed as above, without any additional terms or
conditions.
