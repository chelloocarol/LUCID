# LUCIDMine Method Overview Figure: Structure Summary and Fully Specified Image Prompt

## Code-Grounded Structure Summary

### Naming Constraint
Do not display or emphasize the internal provenance name of the backbone. In the figure, treat the backbone as part of LUCIDMine and label it as:

- `Compact Restoration Encoder`
- `Compressed Restoration Bottleneck`
- `Restoration Decoder`
- `Provisional Restored Image`

Do not use the label `CoA` anywhere in the figure.

### Main Data Flow
Input:

- `Input Mine Conveyor Image I`
- Shape: `H x W x 3`

Main restoration trunk:

- `Input Projection`: `3 -> 8`, right-side output tag `F1: H x W x 8`; this is the explicit input feature to the encoder.
- `Encoder Stage E1`: `8 -> 16`, output `F2: H/2 x W/2 x 16`
- `Encoder Stage E2`: `16 -> 32`, output `F4: H/4 x W/4 x 32`
- `Encoder Stage E3`: `32 -> 64`, output `F8: H/8 x W/8 x 64`
- `Encoder Stage E4`: `64 -> 128`, output `E4: H/16 x W/16 x 128`
- `Compressed Restoration Bottleneck`: receives `E4` and outputs `Z: H/16 x W/16 x 128`
- `VCA`: visibility-conditioned bottleneck adaptation, output `Z_m: H/16 x W/16 x 128`
- `Bottleneck Restoration Block`: residual feature restoration at `128` channels, output `B_m: H/16 x W/16 x 128`
- `Decoder Stage D4`: `128 -> 64`, output `H/8 x W/8 x 64`
- `Decoder Stage D3`: `64 -> 32`, output `H/4 x W/4 x 32`
- `Decoder Stage D2`: `32 -> 16`, output `H/2 x W/2 x 16`
- `Decoder Stage D1`: `16 -> 8`, output `H x W x 8`
- `Output Projection`: `8 -> 3`, output `R: provisional restored image`
- `GARC`: glare-aware residual calibration, output `J: final restored image`

Skip connections:

- `F8 -> D4`
- `F4 -> D3`
- `F2 -> D2`
- `F1 -> D1`

Use thin gray dashed arrows for skip connections. In the overview panel, route all skip connections above the main pipeline, not below it.

### Mine-Prior Branch
The input image is reused by a shared mine-prior extractor. The same prior formulation is invoked at two resolutions. In the overview panel, use one shared prior-extractor group below the main pipeline, not two visually independent networks. Keep all prior arrows in the lowest routing lane, separate from the residual-reference arrow and the skip arrows.

- `Shared Mine Prior Extractor` has two output heads:
- `@ 1/16 scale` feeds VCA.
- `@ full scale` feeds GARC.

Prior tensor:

- `P_m = [Y, D, G_p, V]`
- `Y`: luminance prior
- `D`: dark-channel prior
- `G_p`: glare prior
- `V`: low-visibility prior

Do not place the full formulas in Panel A. If formulas are shown, place them only inside the compact mine-prior inset in Panel B; otherwise use a short caption reference such as `mine priors, see Eqs. (1)-(4)`.

Compact prior formulas for the Panel B inset:

- `Y = 0.299R + 0.587G + 0.114B`
- `D = min(R,G,B)`
- `G_p = clip(max(R,G,B)-Y,0,1)`
- `V = clip(1-|Y-mu_Y|/tau_v,0,1)`

Use `G_p` instead of `G` to avoid confusion with the green channel.

### VCA Detail
VCA = `Visibility-Conditioned Adapter`.

Inputs:

- bottleneck feature `Z: H/16 x W/16 x 128`
- resized mine prior tensor `P_m^{1/16}: H/16 x W/16 x 4`

Internal operations:

- `P_m -> 1x1 Conv 4->128 -> PReLU -> 1x1 Conv 128->128`
- reliability map: `R(P_m) = (0.55 + 0.45V)(1 - 0.35G_p)`
- zero-gated residual injection:
  `Z_m = Z + gamma_v * Phi(P_m) * R(P_m)`

Important visual semantics:

- `gamma_v` is a learnable scalar gate initialized to zero.
- `R(P_m)` is pixel-wise and broadcast to 128 channels.
- channel count remains unchanged: `128 -> 128`.

### GARC Detail
GARC = `Glare-Aware Residual Calibration module`.

Inputs:

- original image `I`
- provisional restored image `R`
- full-resolution mine prior tensor `P_m: H x W x 4`

Internal operations:

- residual: `Delta = R - I`
- prior projection: `P_m -> 1x1 Conv 4->3 -> tanh -> S_l`
- residual mask: `M_g = (1 - G_p)(0.5 + 0.5V)`
- residual scale: `S = clip(1 + gamma_r * S_l * M_g, 0.55, 1.45)`
- final output: `J = I + Delta * S`

Important visual semantics:

- GARC is not a second reconstruction network.
- GARC only calibrates the image-space residual.
- `gamma_r` is a learnable scalar gate initialized to zero.
- residual scale is `H x W x 3`, not a scalar and not a single-channel map.

## Fully Specified Image Prompt

Create a professional publication-quality academic method overview diagram for a computer vision paper.

Title inside the figure:

`LUCIDMine: Visibility-Adaptive and Glare-Calibrated Restoration for Underground Conveyor-Belt Monitoring Images`

This is a method architecture diagram, not a poster and not a generic flowchart. It must show concrete modules, exact labels, tensor sizes, and data-flow arrows. The figure should look like a CVPR/ICCV/IEEE journal method figure: clean white background, readable labels, balanced spacing, and clear grouping.

Critical naming constraint:

Do not write `CoA` anywhere in the figure. Treat the internal restoration trunk as part of LUCIDMine and label it only as `Compact Restoration Backbone`, `Compact Restoration Encoder`, `Compressed Restoration Bottleneck`, and `Restoration Decoder`.

Canvas:

- Landscape layout, suitable for a two-column paper figure.
- Aspect ratio around 16:9 or 2.0:1.
- High resolution, 300 dpi if raster.
- Prefer SVG/PDF style if possible; if raster, ensure all labels are crisp.

Overall layout:

Use a three-panel structure:

- Panel A, top and widest: `A. Overall LUCIDMine Pipeline`
- Panel B, bottom-left: `B. Visibility-Conditioned Adapter (VCA)`
- Panel C, bottom-right: `C. Glare-Aware Residual Calibration (GARC)`

Strict Panel A routing rules:

- Main data flow runs horizontally through the middle of Panel A.
- Skip connections must arc in the top lane above the main pipeline only.
- The residual-reference arrow `I -> GARC` must use a separate upper-middle lane: route it above the main pipeline but below the high skip arcs. Use a thin dark-gray dash-dot line so it cannot be confused with either skip connections or prior guidance.
- Mine-prior branches must stay in the lowest lane below the main pipeline only.
- Maintain at least 8 pt vertical spacing between the upper-middle residual-reference lane and the lower prior-guidance lane.
- Do not let skip arrows, residual-reference arrows, and prior arrows cross.
- Place one shared prior-extractor group below the main pipeline, roughly under the bottleneck-to-output region.
- The shared prior group must be a single larger container labeled `Shared Mine Prior Extractor`; inside it show two small output heads labeled `@ 1/16 Scale` and `@ Full Scale`.
- Route one green/teal input arrow from `Input Mine Conveyor Image I` to this shared prior group in the lowest lane; use a thin solid line with small arrowheads to avoid visual dominance.
- Route the `@ 1/16 Scale` output upward into `VCA`.
- Route the `@ Full Scale` output upward into `GARC`.
- Place the note `same prior formulation, invoked at different spatial scales` inside the shared prior group, centered below the two output heads.
- Add small corner callouts `(B)` on the Panel A VCA block and `(C)` on the Panel A GARC block to anchor the detail panels below.

Panel A: Overall LUCIDMine Pipeline

Arrange the main pipeline from left to right:

1. `Input Mine Conveyor Image I`
   - sublabel: `H x W x 3`
   - small image placeholder or image-like rectangle.
2. `Input Projection`
   - sublabel: `3 -> 8`
   - right-side output tag: `F1: H x W x 8`
   - note: `encoder input`
   - this must be a visible small block before the encoder, not hidden inside the encoder.
3. `Compact Restoration Encoder`
   - show four internal stacked stages:
   - `F2: H/2 x W/2 x 16`
   - `F4: H/4 x W/4 x 32`
   - `F8: H/8 x W/8 x 64`
   - `E4: H/16 x W/16 x 128`
4. `Compressed Restoration Bottleneck`
   - input label: `E4`
   - output label: `Z: H/16 x W/16 x 128`
5. `VCA (B)`
   - full label: `Visibility-Conditioned Adapter`
   - sublabel: `zero-gated prior injection, 128 -> 128`
   - output tag: `Z_m`
6. `Bottleneck Restoration Block`
   - sublabel: `residual feature restoration`
   - output label: `B_m: H/16 x W/16 x 128`
   - visually separate this from VCA: VCA is prior injection; this block is feature restoration.
7. `Restoration Decoder`
   - show four internal stacked stages:
   - `D4: 128 -> 64`
   - `D3: 64 -> 32`
   - `D2: 32 -> 16`
   - `D1: 16 -> 8`
8. `Output Projection`
   - sublabel: `8 -> 3`
9. `Provisional Restored Image R`
   - sublabel: `H x W x 3`
10. `GARC (C)`
   - full label: `Glare-Aware Residual Calibration`
   - sublabel: `image-space residual scaling`
   - include a small internal text inside the block: `Delta = R - I`
11. `Final Restored Image J`
    - sublabel: `H x W x 3`

Main arrows:

- `Input Mine Conveyor Image I -> Input Projection`, label: `RGB image`
- `Input Projection -> Compact Restoration Encoder`, label: `F1`
- `Compact Restoration Encoder -> Compressed Restoration Bottleneck`, label: `E4`
- `Compressed Restoration Bottleneck -> VCA`, label: `Z`
- `VCA -> Bottleneck Restoration Block`, label: `Z_m`
- `Bottleneck Restoration Block -> Restoration Decoder`, label: `B_m`
- `Restoration Decoder -> Output Projection`, label: `H x W x 8`
- `Output Projection -> Provisional Restored Image R`, label: `R`
- `Provisional Restored Image R -> GARC`, label: `R`
- `Input Mine Conveyor Image I -> GARC`, label: `I`, draw this as a thinner dark-gray dash-dot residual-reference arrow routed above the main pipeline but below the high skip arcs; avoid crossing skip or prior arrows.
- `GARC -> Final Restored Image J`, label: `J`

Do not label the `R -> GARC` arrow as `Delta`. `Delta = R - I` is computed inside GARC and must be written inside the GARC block or Panel C, not on the incoming arrow.

Skip connections:

- Add gray dashed curved arrows from encoder stages to matching decoder stages:
- `F8 -> D4`
- `F4 -> D3`
- `F2 -> D2`
- `F1 -> D1`
- Label group: `preserved multi-scale skips`
- Route these skip arrows above the main pipeline as high arcs. They must not pass below the pipeline and must not intersect the prior branches.

Mine-prior branch in Panel A:

From the input image, draw one green/teal prior branch into a single shared prior-extractor group:

- `Input Mine Conveyor Image I -> Shared Mine Prior Extractor`, route this arrow along the lowest lane below the main pipeline.
- The shared prior group contains two internal output heads:
- `@ 1/16 Scale`, output label: `P_m^{1/16} = [Y,D,G_p,V]`, arrow upward into `VCA`, label: `visibility priors`.
- `@ Full Scale`, output label: `P_m = [Y,D,G_p,V]`, arrow upward into `GARC`, label: `glare and visibility priors`.

Place this small text note inside the shared prior group:

`same prior formulation, invoked at different spatial scales`

The note must not overlap any arrow or module.

Panel B: VCA detail

Draw VCA as a small internal computation graph.

Inputs:

- left blue input: `Z`
- sublabel: `H/16 x W/16 x 128`
- green input: `P_m^{1/16}`
- sublabel: `H/16 x W/16 x 4`

Prior projection path:

- `P_m^{1/16} -> 1x1 Conv 4->128 -> PReLU -> 1x1 Conv 128->128 -> Phi(P_m)`

Reliability path:

- `P_m^{1/16} -> Reliability Map R(P_m)`
- write compact formula:
  `R(P_m)=(0.55+0.45V)(1-0.35G_p)`
- indicate it is pixel-wise and broadcast to 128 channels.

Fusion:

- show a multiplication node `x` combining `Phi(P_m)` and `R(P_m)`
- show a purple scalar gate node `gamma_v`
- label: `learnable gate, init 0`
- show residual addition node `+` with original `Z`
- output: `Z_m`
- formula near the output:
  `Z_m = Z + gamma_v Phi(P_m) R(P_m)`

Panel C: GARC detail

Draw GARC as an image-space residual scaling module.

Inputs:

- `Original Image I`
- `Provisional Restored Image R`
- `P_m = [Y,D,G_p,V]`

Residual path:

- `R - I -> Delta`
- label: `restoration residual`

Prior scale path:

- `P_m -> 1x1 Conv 4->3 -> tanh -> S_l`
- label: `learned scale, H x W x 3`

Mask path:

- `P_m -> Glare-Visibility Mask M_g`
- formula:
  `M_g=(1-G_p)(0.5+0.5V)`
- label: `suppresses lamp glare, keeps low-visibility correction`

Scale and output:

- combine `S_l`, `M_g`, and purple scalar gate `gamma_r`
- label: `gamma_r init 0`
- show:
  `S = clip(1 + gamma_r S_l M_g, 0.55, 1.45)`
- show multiplication node:
  `Delta x S`
- final addition:
  `J = I + Delta x S`
- output: `Final Restored Image J`

Color and visual style:

- Clean white background.
- Main restoration backbone: low-saturation blue or blue-gray fills, e.g. `#DBEAFE`, border `#2563EB`.
- VCA module: purple fill, e.g. `#EDE9FE`, border `#7C3AED`.
- GARC module: light neutral gray fill, e.g. `#E5E7EB`, border `#EA580C`; the orange border carries the semantic cue while the fill stays grayscale-safe.
- Mine-prior extractor and prior branches: medium-light gray-green fill, e.g. `#B7D8C5`, border `#047857`; this must be visibly darker than the blue backbone fill in grayscale.
- Skip connections: light gray dashed arrows.
- Main arrows: dark gray or black, thick strokes with clear arrowheads.
- Prior arrows: green/teal, thick enough to read.
- Residual/gate nodes: small circles with `+`, `x`, `gamma_v`, `gamma_r`.
- Gate nodes `gamma_v` and `gamma_r` must use dark fill, e.g. `#312E81` or black, with white text and a clear border. Do not use pale purple gate nodes on a pale purple panel.
- Use rounded rectangles, flat solid fills, and thin professional borders.
- Do not use gradient fills; use pure flat colors only.
- Avoid heavy shadows, 3D effects, clipart, cartoon icons, glowing effects, and decorative backgrounds.

Line hierarchy:

- Main pipeline arrows: 2.5 pt, solid, dark gray or black, largest arrowheads.
- Prior branch arrows: 1.3 pt, solid, teal/green, medium arrowheads, routed in the lowest lane.
- Residual reference arrow `I -> GARC`: 1.2 pt, dash-dot dark gray, smaller arrowhead, labeled `I`, routed in the upper-middle lane above the main pipeline and below skip arcs.
- Skip connections: 1.0 pt, dashed, light gray, smallest arrowheads, routed above the main pipeline.
- Use both line width and line style to separate flow types; do not rely on color alone.

Typography:

- Use English labels only.
- Use sans-serif font such as Arial or Helvetica.
- Main module labels should be bold.
- Minimum readable label size equivalent to 14 pt in a two-column paper.
- Do not use long paragraphs inside the figure.

Arrow requirements:

- Every arrow must have a clear source and target.
- No arrow crossings.
- Main data flow must be visually readable in under five seconds.
- Arrowheads must be large and unambiguous.
- Use labels on important arrows: `RGB image`, `F1`, `E4`, `Z`, `Z_m`, `B_m`, `H x W x 8`, `R`, `I`, `J`, `P_m`.
- Do not place `Delta` on any incoming Panel A arrow. `Delta = R - I` belongs inside GARC.

Do not include:

- The word `CoA`.
- RIDCP or any comparison method.
- Transformer, attention, diffusion, prompt learning, codebook, or any module not present in the current LUCIDMine implementation.
- A separate second decoder for GARC.
- A global scalar reliability map; the reliability map must be pixel-wise.
- A single-channel residual scale; residual scale is image-space and three-channel.

Final visual goal:

The diagram should communicate that LUCIDMine is a compact underground conveyor-image restoration framework with two explicit additions:

1. VCA changes the compressed representation using mine visibility priors.
2. GARC changes the final output behavior by calibrating image-space residuals around glare and low visibility.

## Layout Review Fixes Applied

This prompt has been revised according to the first and second layout/style reviews:

- Input Projection is now an explicit visible block.
- The `F1` label is placed at the Input Projection output and explicitly marked as the encoder input.
- The encoder terminal stage is labeled `E4`, while `Z` is reserved for the compressed bottleneck tensor to avoid duplicate tensor naming.
- Bottleneck Restoration Block is separated from VCA and labeled as feature restoration with output size.
- Prior extraction is now shown as one shared extractor group with two scale output heads, reducing duplicate long guidance lines.
- The prior-scale note is placed inside the shared prior group.
- Skip connections are fixed in the upper lane above the main pipeline.
- The `I -> GARC` residual-reference arrow is routed in a separate upper-middle lane with dash-dot styling.
- Line hierarchy is specified by width and style, not only color.
- Panel A VCA/GARC blocks include `(B)` and `(C)` anchors to their detail panels.
- Panel A includes a direct `I -> GARC` residual-reference arrow.
- The `R -> GARC` arrow is labeled only `R`.
- `Delta = R - I` is placed inside GARC, not on an input arrow.
- The Panel C defensive note has been removed from the figure prompt; that explanation belongs in the caption or main text.
- Full prior formulas are prohibited in Panel A and restricted to Panel B or caption equation references.
- Ambiguous arrow labels were changed from functional phrases to tensor labels, e.g. `H x W x 8` and `J`.
- Module fills now use grayscale-safe flat colors; gradients are explicitly disallowed.
- Gate nodes use dark fill with white text for grayscale readability.
