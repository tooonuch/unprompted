> 💡 **Notice:** Shortcode documentation has moved! It is now available inside of the Wizard panel of the WebUI. [You may also view the individual markdown files on GitHub.](https://github.com/ThereforeGames/unprompted/tree/main/docs/shortcodes)

# Unprompted Manual

Shortcode syntax is subject to change based on community feedback.

If you encounter any confusing, incomplete, or out-of-date information here, please do not hesitate to open an issue. I appreciate it!

## ❔ Known Issues

<details><summary>WebUI slowdowns</summary>

Due to the nature of Gradio, creating many UI elements or event listeners leads to performance issues in the WebUI. This may be resolved in Gradio 4, as [suggested here](https://github.com/gradio-app/gradio/issues/4841#issuecomment-1632141732).

In the meantime, you can improve performance by disabling Wizard tabs you do not use. For example, you can disable the Shortcodes tab by setting `ui.wizard_shortcodes` to false in `config_user.json`.

</details>

<details><summary>Compatibility with ControlNet</summary>

To achieve compatibility between Unprompted and ControlNet, you must manually rename the `unprompted` extension folder to `_unprompted`. This is due to [a limitation in the Automatic1111 extension framework](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8011) whereby priority is determined alphabetically.

Additionally, if you're using the Forge WebUI, you should move `_unprompted` to `extensions-builtin/_unprompted` so that it can execute ahead of Forge's native ControlNet extension.

</details>

<details><summary>Compatibility with other extensions</summary>

The following extension(s) are known to cause issues with Unprompted:

- **adetailer**: reportedly utilizes its own prompt field(s) that do not receive Unprompted strings correctly
- **Regional Prompter**: This extension throws an error while processing images in the `[after]` block, however the error does not seem to interfere with the final result and is likely safe to disregard.
- **ControlNet**: To my knowledge, it is not possible to unhook ControlNet in the [after] routine. Please check [this issue](https://github.com/Mikubill/sd-webui-controlnet/issues/2082) for more details.
</details>

<details><summary>A1111 "Lora/Networks: Use Old Method"</summary>

The WebUI setting "Lora/Networks: use old method [...]" is not compatible with Unprompted and will cause a crash during image generation.

</details>

## 🧙 The Wizard

<details><summary>What is the Wizard?</summary>

The Unprompted WebUI extension has a dedicated panel called the Wizard. It is a GUI-based shortcode builder.

Pressing **"Generate Shortcode"** will assemble a ready-to-use block of code that you can add to your prompts.

Alternatively, you can enable `Auto-include this in prompt` which will add the shortcode to your prompts behind the scenes. This essentially lets you use Unprompted shortcodes as if they were standalone scripts. You can enable/disable this setting on a per-shortcode basis.

The Wizard includes three distinct modes: Shortcodes, Templates, and Capture.

</details>

<details><summary>Shortcodes Mode</summary>

This mode presents you with a list of all shortcodes that have a `ui()` block in their source code.

You can add Wizard UI support to your own custom shortcodes by declaring a `ui()` function as shown below:

```
	def ui(self,gr):
		gr.Radio(label="Mask blend mode 🡢 mode",choices=["add","subtract","discard"],value="add",interactive=True)
		gr.Checkbox(label="Show mask in output 🡢 show")
		gr.Checkbox(label="Use legacy weights 🡢 legacy_weights")
		gr.Number(label="Precision of selected area 🡢 precision",value=100,interactive=True)
		gr.Number(label="Padding radius in pixels 🡢 padding",value=0,interactive=True)
		gr.Number(label="Smoothing radius in pixels 🡢 smoothing",value=20,interactive=True)
		gr.Textbox(label="Negative mask prompt 🡢 negative_mask",max_lines=1)
		gr.Textbox(label="Save the mask size to the following variable 🡢 size_var",max_lines=1)
```

The above code is the entirety of txt2mask's UI at the time of writing. We recommend examining the .py files of other shortcodes if you want to see additional examples of how to construct your UI.

Every possible shortcode argument is exposed in the UI, labeled in the form of `Natural description 🡢 technical_argument_name`. The Wizard only uses the technical_argument_name when constructing the final shortcode.

There are a few reserved argument names that will modify the Wizard's behavior:

- `arg_verbatim`: This will inject the field's value directly into the shortcode. Useful for shortcodes that can accept multiple, optional arguments that do not have pre-determined names.
- `arg_str`: This will inject the field's value into the shortcode, enclosing it in quotation marks.
- `arg_int`: This will inject the field's value into the shortcode, casting it as an integer. 

Note that every `technical_argument_name` must be unique, so if you need to re-use one of the above reserved names, you can append a number to it. For example, `arg_verbatim_1`, `arg_verbatim_2`, etc.

</details>

<details><summary>Templates Mode</summary>

This mode presents you with a list of txt files inside your `Unprompted/templates` directory that begin with a `[template]` block.

By including this block in your file, Unprompted will parse the file for its `[set x _new]` statements and adapt those into a custom Wizard UI.

The `_new` argument means *"only set this variable if it doesn't already exist,"* which are generally the variables we want to show in a UI.

The `[template]` block supports the optional `name` argument which is a friendly name for your function shown in the Templates dropdown menu.

The content of `[template]` is a description of your script to be rendered with [Markdown](https://www.markdownguide.org/basic-syntax/), which means you can include rich content like pictures or links. It will show up at the top of your UI.

The `[set]` block supports `_ui` which determines the type of UI element to render your variable as. Defaults to `textbox`. Here are the possible types:

- `textbox`: Ideal for strings. The content of your `[set]` block will be rendered as placeholder text.
- `number`: Ideal for integers. 
- `radio`: A list of radio buttons that are determined by the `_choices` argument, constructed as a delimited list.
- `dropdown`: A dropdown menu that is populated by the `_choices` argument, constructed as a delimited list.
- `slider`: Limits selection to a range of numbers. You must also specify `_minimum`, `_maximum` and `_step` (step size, normally 1) for this element to work properly.
- `none`: Do not create this block as a UI element even if it has the `_new` parg.

The `[set]` block supports `_label` which is the friendly text to use above the UI element. If not set, the label will default to the variable name you're calling with `[set]` in titlecase format (e.g. "my_variable" becomes "My Variable.")

The `[set]` block supports `_show_label` which lets you toggle visibility of the label in the UI. Defaults to True.

The `[set]` block supports `_info` which is descriptive text that will appear near the UI element.

The `[set]` block supports `_lines` and `_max_lines` to specify the number of rows shown in a `textbox` element.

Supports the `[wizard]` shortcode which will group the inner `[set]` blocks into a group UI element, the type of which is defined by the first parg: `accordion`, `row`, or `column`.

</details>

<details><summary>Capture Mode</summary>

This mode offers a convenient way to produce the code for the last image you generated.

It has a few settings that change how the code is formatted:

- **Include inference settings:** Determines which inference options to show as a `[sets]` block. These are settings such as CFG Scale, batch count, etc. On `simple`, it will exclude variables with a value of 0 as well as empty strings. `Verbose` gives you everything.
- **Include (negative) prompt:** Determines whether to show the prompt. On `original`, it will show the prompt with shortcodes intact, whereas `postprocessed` gives you the prompt after shortcodes have been executed.
- **Include model:** adds the checkpoint name to the `[sets]` block.
- **Add [template] block**: Prepends the result with a placeholder `[template]` block that makes your code compatible with the Wizard Templates tab.


</details>

## 🎓 Proficiency

<details><summary>Atomic vs Block Shortcodes</summary>

Unprompted supports two types of shortcodes:

- Block shortcodes that require an end tag, e.g. `[set my_var]This is a block shortcode[/set]`
- Atomic shortcodes that are self-closing, e.g. `[get my_var]`

These are mutually exclusive. Shortcodes must be defined as one or the other.

</details>

<details><summary>Secondary Shortcode Tags</summary>

You can use shortcodes directly in the arguments of other shortcodes with **secondary tags.**

To do this, simply write your tags with squiggly brackets `{}` instead of square brackets `[]`. Let's look at an example:

```
[file "{choose}some_file|another_file{/choose}"]
```

Instead of passing `[file]` a specific filename, we are using `{choose}` to pick between `some_file` and `another_file`.

Secondary shortcode tags can have infinite nested depth. The number of `{}` around a shortcode indicates its nested level.

> **💡 Notice:** As of Unprompted v9.0.0, do not use secondary shortcode tags inside of block content. They are only used within arguments now. Please see the 25 April 2023 announcement for more information.

</details>

<details><summary>Advanced Expressions</summary>

Most shortcodes support programming-style evaluation via the [simpleeval library](https://github.com/danthedeckie/simpleeval).

This allows you to enter complex expressions in ways that would not be possible with standard shortcode arguments. For example, the `[if]` shortcode expects unique variable keys and a singular type of comparison logic, which means you **cannot** do something like this:

`[if var_a>=1 var_a!=5]`

However, with advanced expressions, you definitely can! Simply put quotes around your expression and Unprompted will parse it with simpleeval. Check it out:

`[if "var_a>=10 and var_a!=5"]Print me[/if]`

If you wish to compare strings, use `is` and single quotes as shown below:

`[if "var_a is 'man' or var_a is 'woman'"]My variable is either man or woman[/if]`

You can even mix advanced expressions with shortcodes. Check this out:

`[if "var_a is {file test_one} or var_a is {choose}1|2|3{/choose}"]`

**The secondary shortcode tags are processed first** and then the resulting string is processed by simpleeval.

For more information on constructing advanced expressions, check the documentation linked above.

</details>

<details><summary>Escaping Characters</summary>

Use the backtick to print a character as a literal part of your prompt. This may be useful if you wish to take advantage of the prompt editing features of the A1111 WebUI (which are denoted with square brackets and could thus conflict with Unprompted shortcodes.)

Note: if a shortcode is undefined, Unprompted will print it as a literal as if you had escaped it.

```
Photo of a `[cat|dog`]
```

</details>

<details><summary>Multi-line Atomic Shortcodes</summary>

In my testing, it appears to be possible to write lengthy atomic shortcodes across multiple lines, provided **the first line has a trailing space** and **each subsequent line has a leading space.** Here is an example:

```
[sets 
 cn_0_enabled=1 cn_0_pixel_perfect=1 cn_0_module=softedge_hed cn_0_model=controlnet11Models_softedge cn_0_weight=0.25
 cn_1_enabled=1 cn_1_pixel_perfect=1 cn_1_module=mediapipe_face cn_1_model=control_mediapipe_face_sd15_v2 cn_1_weight=1.0
 cn_2_enabled=1 cn_2_pixel_perfect=1 cn_2_module=t2ia_color_grid cn_2_model=coadapter-color-sd15v1 cn_2_weight=1.0
 cn_3_enabled=1 cn_3_pixel_perfect=1 cn_3_module=openpose_full cn_3_model=controlnet11Models_openpose cn_3_weight=1.0
 ]
```

Do note, however, that your mileage may vary if you are not using the default sanitization rules.

</details>

<details><summary>Special Variables</summary>

In addition to all of the Stable Diffusion variables exposed by Automatic1111's WebUI, Unprompted gives you access to the following variables:

### global variables

These variables are loaded from your `config.json` file and made available to all shortcodes and templates.

They are prefixed with `Unprompted.Config.syntax.global_prefix`, which defaults to `global_`. Therefore, if you have a variable called `subject` in your config file, you can access it with `[get global_subject]`.

The content of a global variable is processed when selected with `[get]`, which means you can store complex values into these variables such as functions or shortcodes, and they will not impact performance until you actually retrieve the value.

### batch_count_index

An integer that correponds to your progress in a batch run. For example, if your batch count is set to 5, then `batch_count_index` will return a value from 0 to 4.

**Note:** This was formerly known as `batch_index`, which still works but is considered deprecated due to its lack of specificity. It may be removed from a future update.

### batch_size_index

An integer that corresponds to your progress within a specific batch. For example, if your batch size is set to 5, then `batch_size_index` will return a value from 0 to 4.

### batch_test

Shortcodes that implement batch processing--such as `[zoom_enhance]`--will test the expression stored in `batch_test` against the batch item index to determine if an image should be bypassed by the shortcode.

Example: you set `batch_test` to `<= 3` and you run a batch process with 5 images. The fifth image will be skipped by shortcodes such as `[zoom_enhance]`. (`batch_index` is zero-indexed, so 3 corresponds to the fourth image.)

### default_image

In the event that a shortcode such as `[zoom_enhance]` is unable to determine which image it should process, it will fallback to this filepath variable instead of throwing an error.

### sd_model

You can set this variable to the name of a Stable Diffusion checkpoint, and Unprompted will load that checkpoint at the start of inference. This variable is powered by the WebUI's `get_closet_checkpoint_match()` function, which means that your model name does not have to be 100% accurate - but you should strive to use a string that's as accurate as possible.

### sd_base

This variable contains the base type of the currently-loaded checkpoint. The possible values at the time of writing are as follows:

- sdxl
- sd3
- sd2
- sd1
- none

When you load a new model via `sd_model`, Unprompted will change the value of `sd_base` accordingly. Due to this, it is not recommended that you set the value of `sd_base` by hand; you can think of it as a soft-read-only variable.

### sd_vae

Similar to the `sd_model` variable, you can load a VAE by setting to this a filename sans extension.

```
[sets sd_vae="vae-ft-mse-840000-ema-pruned"]
```

### single_seed

You can set this integer variable to lock the seed for all images in a batch run.

You might be wondering why `[set seed]x[/set]` won't accomplish the same thing. Well, this is because the WebUI populates another variable called `all_seeds` for batch runs. It takes your initial seed (let's say 1000) and increments it by 1 for each successive image. So if you're making 4 images (i.e. `batch_count` = 4) and you set `seed` to 1000, your seeds will actually look like this: 1000, 1001, 1002, 1003.

The `single_seed` variable will instead force all the seeds to your chosen value.

It is functionally equivalent to the following:

```
[array all_seeds _fill=1000]
```

### controlnet_*

You can use `[set]` to manage ControlNet settings in this format:

```
[sets controlnet_unit_property=value]
```

Where **unit** is an integer that corresponds to the index of a ControlNet unit (between 0 and your maximum number of units).

Here is a list of valid properties at the time of writing:

- enabled
- module
- model
- weight
- image (loads a file from a filepath string)
- invert_image
- resize_mode
- rgbbgr_mode
- low_vram
- processor_res
- threshold_a
- threshold_b
- guidance_start
- guidance_end
- guess_mode

For example, we can enable units #0 and #3 and set the weight of unit #3 to 0.5 as follows:

```
[sets controlnet_0_enabled=1 controlnet_3_enabled=1 controlnet_3_weight=0.5]
```

You may also use the shorthand `cn_` in place of `controlnet_`.

Due to the WebUI's extension architecture, all images in a batch are processed by Unprompted before ControlNet, meaning it is not possible to update `cn_` in the middle of a batch run.

</details>

<details><summary>Why some shortcode arguments begin with an _underscore</summary>

We use underscores to denote optional system arguments in shortcodes that may also accept dynamic, user-defined arguments.

Take a look at `[replace]` as an example.

`[replace]` allows you to modify a string with arbitrary before-after argument pairings, e.g. `[replace this=that red=blue]`.

However, `[replace]` also features system arguments like `_count` and so the shortcode must have a way to differentiate between the two types.

In short, if the argument begins with `_`, the program will assume it is a system argument of some kind.

That said, we're still ironing out the methodology for underscores - at the moment, some arguments may use underscores where it isn't strictly necessary. If you find any such cases feel free to open an Issue or Discussion Thread about it.

</details>

<details><summary>The config file</summary>

Various aspects of Unprompted's behavior are controlled through `unprompted/lib_unprompted/config.json`.

If you wish to override the default settings, you should make another file at the same location called `config_user.json`. Modifications to the original config file will **not** be preserved between updates.

Here are some of the settings you can modify:

<details><summary>beta_features (bool)</summary>

Opt into unfinished features by setting `beta_features` to true.

</details>

<details><summary>skip_requirements (bool)</summary>

Setting this to true will bypass the Python dependencies check at startup, allowing the WebUI to load more quickly. If you use this, please remember to disable it between major updates of Unprompted.

Also note that this setting **must** be specified in `config_user.json` - it has no effect in `config.json`.

</details>

<details><summary>advanced_expressions (bool)</summary>

This determines whether expressions will be processed by simpleeval. Disable for slightly better performance at the cost of breaking some templates.

</details>

<details><summary>template_directory (str)</summary>

This is the base directory for your text files.

</details>

<details><summary>stable_diffusion.batch_count_method (str)</summary>

Determines how Unprompted will process images when `batch_count` > 1.

The default method is `standard` which utilizes the WebUI's `process_batch()` routine to evaluate your prompt before each image generation.

Supports `safe` method which pre-processes all images at the beginning of the batch run. This method prevents system variables such as CFG scale or model checkpoint from being altered mid-run but may have better compatibility with some shortcodes or extensions.

Supports `unify` method which causes all images in a batch run to have the same settings as the first image.

</details>

<details><summary>stable_diffusion.batch_size_method (str)</summary>

Determines how Unprompted will process images when `batch_size` > 1.

The default method is `standard` which evaluates the prompt before each image generation.

Supports `unify` method which causes all images in one batch to have the same prompt settings.

</details>

<details><summary>syntax.sanitize_before (dict)</summary>

This is a dictionary of strings that will be replaced at the start of processing. By default, Unprompted will swap newline and tab characters to the `\\n` placeholder.

</details>

<details><summary>syntax.sanitize_after (dict)</summary>

This is a dictionary of strings that will be replaced after processing. By default, Unprompted will convert the `\\n` placeholder to a space.

</details>

<details><summary>syntax.tag_start (str)</summary>

This is the string that indicates the start of a shortcode.

</details>

<details><summary>syntax.tag_end (str)</summary>

This is the string that indicates the end of a shortcode.

</details>

<details><summary>syntax.tag_start_alt (str)</summary>

This is the string that indicates the start of a secondary shortcode.

</details>

<details><summary>syntax.tag_end_alt (str)</summary>

This is the string that indicates the end of a secondary shortcode.

</details>

<details><summary>syntax.tag_close (str)</summary>

This is the string that indicates the closing tag of a block-scoped shortcode.

</details>

<details><summary>syntax.tag_escape (str)</summary>

This is the string that allows you to print a shortcode as a literal string, bypassing the shortcode processor.

Note that you only have to include this string once, before the shortcode, as opposed to in front of every bracket.

</details>

<details><summary>templates.default (str)</summary>

This is the final string that will be processed by Unprompted, where `*` is the user input.

The main purpose of this setting is for hardcoding shortcodes you want to run every time. For example: `[img2img_autosize]*`

</details>

<details><summary>templates.default_negative (str)</summary>

Same as above, but for the negative prompt.

</details>

</details>

## 👨‍💻 For Programmers

<details><summary>Creating Your Own Shortcodes</summary>

Shortcodes are loaded as Python modules from `unprompted/shortcodes`. You can make your own shortcodes by creating files there (preferably within a subdirectory called `custom`.)

The shortcode name is defined by the filename, e.g. `override.py` will give you the ability to use `[override]`. Shortcode filenames should be unique.

A shortcode is structured as follows:

```
class Shortcode():
	"""A description of the shortcode goes here."""
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted

	def run_block(self, pargs, kwargs, context,content):
		
		return("")

	def cleanup(self):
		
		return("")
```

You can declare an atomic shortcode by replacing `run_block()` with `run_atomic()`:

```
def run_atomic(self, pargs, kwargs, context):
```

Unlike Blocks, our Atomic shortcode does not receive a `content` variable.

The `__init__` function gives the shortcode access to our main Unprompted object, and it's where you should declare any unique variables for your shortcode.

The `run_block` function contains the main logic for your shortcode. It has access to these special variables (the following list was pulled from the [Python Shortcodes](https://www.dmulholl.com/dev/shortcodes.html) library, on which Unprompted depends):

- `pargs`: a list of the shortcode's positional arguments.
- `kwargs`: a dictionary of the shortcode's keyword arguments.
- `context`: an optional arbitrary context object supplied by the caller.
- `content`: the string within the shortcode tags, e.g. `[tag]content[/tag]`

Positional arguments (`pargs`) and keyword arguments (`kwargs`) are passed as strings. The `run_` function itself returns a string which will replace the shortcode in the parsed text.

The `cleanup()` function runs at the end of the processing chain. You can free any unnecessary variables from memory here.

Regarding Blocks, it is important to understand that **the parser evalutes inner shortcodes before outer shortcodes.** Sometimes this is not desirable, such as when dealing with a "conditional" shortcode like `[if]`. Let's consider the following example:

```
[if my_var=1][set another_var]0[/set][/if]
```

In this case, we **do not** want to set `another_var` to 0 unless the outer `[if]` statement succeeds. For this reason, the `[if]` shortcode includes a special `preprocess_block()` function:

```
def preprocess_block(self,pargs,kwargs,context): return True
```

When the parser encounters a block shortcode, it runs the `preprocess_block()` function if it exists. If that function returns True, then any future shortcodes are temporarily blocked by the parser until it finds the endtag (`[/if]`). This is what allows us to override the normal "inner-before-outer" processing flow.

The `preprocess_block()` function is also useful for executing arbitrary code before parsing the remaining text. Just be aware that the function is not aware of the shortcode's content, and that no `run_...()` functions have executed before this step.

</details>

<details><summary>Implementing support for [else]</summary>

In most programming languages, the "else" statement is joined at the hip with "if." However, thanks to the modular nature of Unprompted, we can use "else" with a variety of blocks and they do not even have to be placed next to each other.

One such example is `[chance]`; you can follow a statement like `[chance 30]` with `[else]` to catch the 70% of cases where the chance fails.

As of Unprompted v9.14.0, any shortcode can implement full compatibility with `[else]` in just a few lines of code. Here's how:

1. Conditional shortcodes need to instantiate the `preprocess_block()` method in order to prevent execution of content unless the condition evaluates to true.

```
def preprocess_block(self, pargs, kwargs, context):
	return True
```

2. Now in the `run_block()` method, immediately after testing our condition and finding that it's true, you must call `self.Unprompted.prevent_else()` as shown:

```
if some_condition:
	self.Unprompted.prevent_else(else_id)
```

This will tell the `[else]` block not to execute at the current conditional depth level. It also increments our depth level by 1 (`self.Unprompted.conditional_depth += 1`) to account for the possibility of further if/else-type statements in the content.

You should also define `else_id` near the top of your `run_block()` like this:

```
else_id = kwargs["_else_id"] if "_else_id" in kwargs else str(self.Unprompted.conditional_depth)
```

The `else_id` is a string variable that defaults to our conditional depth. By letting the user specify a custom `else_id`, they can tie the "if" statement to a specific `[else]` block anywhere in the script. 

3. On the other hand, if our statement evalutes to false, we need to give `[else]` the green light:

```
else: self.Unprompted.shortcode_objects["else"].do_else[else_id] = True
```

4. Finally, just before the return statement, we must reset the conditional depth to 0:

```
self.Unprompted.conditional_depth = 0
return some_value
```

And you're set!



</details>