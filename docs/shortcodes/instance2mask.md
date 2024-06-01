Uses Mask R-CNN (an instance segmentation model) to predict instances. The found instances are mask. Different from `[txt2mask]` as it allows to run the inpainting for each found instance individually. This is useful, when using high resolution inpainting. This shortcode only works in the img2img tab of the A1111 WebUI.
**Important:** If per_instance is used it is assumed to be the last operator changing the mask.

The supported classes of instances are:
- `person`
- `bicycle`
- `car`
- `motorcycle`
- `airplane`
- `bus`
- `train`
- `truck`
- `boat`
- `traffic light`
- `fire hydrant`
- `stop sign`
- `parking meter`
- `bench`
- `bird`
- `cat`
- `dog`
- `horse`
- `sheep`
- `cow`
- `elephant`
- `bear`
- `zebra`
- `giraffe`
- `backpack`
- `umbrella`
- `handbag`
- `tie`
- `suitcase`
- `frisbee`
- `skis`
- `snowboard`
- `sports ball`
- `kite`
- `baseball bat`
- `baseball glove`
- `skateboard`
- `surfboard`
- `tennis racket`
- `bottle`
- `wine glass`
- `cup`
- `fork`
- `knife`
- `spoon`
- `bowl`
- `banana`
- `apple`
- `sandwich`
- `orange`
- `broccoli`
- `carrot`
- `hot dog`
- `pizza`
- `donut`
- `cake`
- `chair`
- `couch`
- `potted plant`
- `bed`
- `dining table`
- `toilet`
- `tv`
- `laptop`
- `mouse`
- `remote`
- `keyboard`
- `cell phone`
- `microwave`
- `oven`
- `toaster`
- `sink`
- `refrigerator`
- `book`
- `clock`
- `vase`
- `scissors`
- `teddy bear`
- `hair drier`
- `toothbrush`

Supports the `mode` argument which determines how the text mask will behave alongside a brush mask:
- `add` will overlay the two masks. This is the default value.
- `discard` will ignore the brush mask entirely.
- `subtract` will remove the brush mask region from the text mask region.
- `refine` will limit the inital mask to the selected instances.

Supports the optional `mask_precision` argument which determines the confidence of the instance mask. Default is 0.5, max value is 1.0. Lowering this value means you may select more than you intend per instance (instances may overlap).

Supports the optional `instance_precision` argument which determines the classification thresshold for instances to be masked. Reduce this, if instances are not detected successfully. Default is 0.85, max value is 1.0. Lowering this value can lead to wrongly classied areas.

Supports the optional `padding` argument which increases the radius of the instance masks by a given number of pixels.

Supports the optional `smoothing` argument which refines the boundaries of the mask, allowing you to create a smoother selection. Default is 0. Try a value of 20 or greater if you find that your masks are blocky.

Supports the optional `select` argument which defines how many instances to mask. Default value is 0, which means all instances.

Supports the optional `select_mode` argument which specifies which instances are selected:
- `overlap` will select the instances starting with the instance that has the greatest absolute brushed mask in it.
- `overlap relative` behaves similar to `overlap` but normalizes the areas by the size of the instance.
- `greatest area` will select the greatest instances by pixels first.
- `random` will select instances in a random order
Defaults to `overlap`.

Supports the optional `show` positional argument which will append the final masks to your generation output window and for debug purposes a combined instance segmentation image.

Supports the optional `per_instance` positional argument which will render and append the selected masks individually. Leading to better results if full resolution inpainting is used.

```
[instance2mask]clock[/txt2mask]
```