import numpy as np
import config

def make_mask(processing_size):
    if config.masking_border is None:
        return None
    
    mask = np.ones(processing_size[::-1], dtype=np.uint8)
    mask[: , : config.masking_border[0]] = 0 # mask left border
    mask[: , -config.masking_border[2] : ] = 0 # mask right border
    mask[ : config.masking_border[1], :] = 0 # mask top
    mask[-config.masking_border[3] :, :] = 0 # mask bottom
    return mask

def scale_proportional(shape, new_width):
    if not new_width:
        return shape[:2] # do not rescale
    
    width = new_width
    f = width / shape[1]
    height = int(f * shape[0])
    return (width, height)