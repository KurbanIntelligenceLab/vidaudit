# Vendored from ljh0v0/FVMD-frechet-video-motion-distance (the `fvmd` package), used by
# the FVMD detector wrapper. The upstream __init__ imports keypoint_tracking, which pulls
# cv2/skimage/tensorflow; the FVMD wrapper only needs nets.pips2 + extract_motion_features
# + utils.{basic,samp,misc}, so this __init__ is intentionally empty to avoid that chain.
