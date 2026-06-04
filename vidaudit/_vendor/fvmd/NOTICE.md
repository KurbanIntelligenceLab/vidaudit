# Vendored: FVMD / PIPs++

This directory vendors code from the FVMD (Fréchet Video Motion Distance) project,
which builds on the PIPs++ persistent point tracker:

- FVMD — https://github.com/ljh0v0/FVMD-frechet-video-motion-distance
- PIPs++ / PointOdyssey — https://github.com/aharley/pips2

It is included largely unmodified, except for a device-agnostic patch in
`nets/pips2.py` (a hard-coded `.cuda()` replaced with the input tensor's device) so
the tracker runs on CPU / MPS / CUDA. The code retains its upstream license; see the
source repositories above for the exact terms.
