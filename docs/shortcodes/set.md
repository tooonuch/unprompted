Sets a variable to the given content.

`_append` will instead add the content to the end of the variable's current value, e.g. if `my_var` equals "hello" then `[set my_var _append] world.[/set]` will make it equal "hello world."

`_prepend` will instead add the content to the beginning of the variable's current value.

Supports the optional `_new` parg which will bypass the shortcode if the variable you're trying to `[set]` already exists. For example:

```
[set my_var]apple[/set]
[set my_var _new]orange[/set]
[get my_var]
```

This example will return `apple`.

Supports the optional `_choices` argument, which is a delimited string of accepted values. The behavior of this argument depends on whether or not the `_new` argument is present:

- If `_new` and the variable exists as a value that is **not** accepted by `_choices`, then `_new` is ignored and this shortcode will run.
- If **not** `_new` and we're trying to set the variable to a value that is **not** accepted by `_choices`, then the `[set]` block is bypassed.
- In the Wizard UI for certain kinds of elements, `_choices` is used to populate the element, such as a dropdown menu or radio group.

Supports all Stable Diffusion variables that are exposed via Automatic's Script system, e.g. `[set cfg_scale]5[/set]` will force the CFG Scale to be 5 for the run.

Supports the `_remember` parg that will invoke the `[remember]` shortcode with your variable. See `[remember]` for more information.

Supports the `_external` kwarg to write the variable to an external .json file. If the file does not exist, it will be created for you.

Supports the `_defer` parg to delay the processing of the content until you call the variable with `[get _parse]`.

```
[set my_var]This is the value of my_var[/set]
```