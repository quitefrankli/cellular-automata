# Training targets

`circle.png` is the project's synthetic starter target.

The following transparent 128×128 PNGs come from Google's Noto Emoji release **v2.047**. They add outlines, multiple colors, asymmetry, and fine features beyond the circle:

- [lizard.png](lizard.png): https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/128/emoji_u1f98e.png
- [butterfly.png](butterfly.png): https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/128/emoji_u1f98b.png
- [octopus.png](octopus.png): https://raw.githubusercontent.com/googlefonts/noto-emoji/v2.047/png/128/emoji_u1f419.png

Copyright Google Inc. Distributed under Apache License 2.0; see [NOTO-LICENSE.txt](NOTO-LICENSE.txt). These files are unmodified upstream assets. Attribution is recorded here; the surrounding project does not inherit this asset license.

Training downsamples each image to a maximum 40×40 footprint on the default 64×64 grid. Inspect the processed `target.png` before judging a trained model: small features may disappear at that resolution. Use each target with a separate model and output folder.
