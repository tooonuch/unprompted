Returns the value of `variable`.

Supports variable parsing with the optional `_var` argument, e.g. `[get _var="somevar"]`.

You can add `_before` and `_after` content to your variable. This is particularly useful for enclosing the variable in escaped brackets, e.g. `[get my_var _before=[ _after=]]` will print `[value of my_var]`.

Supports the optional `_default` argument, the value of which is returned if your variable does not exist e.g. `[get car_color _default="red"]`.

Supports returning multiple variables, e.g. `[get var_a var_b]` will return the values of two variables separated by a comma and space.

You can change the default separator with `_sep`.

Supports the `_external` kwarg to retrieve variable(s) from an external .json file. If the file does not exist, it will be created for you. Please be aware that using `_external` will take precedence over the variable(s) currently stored in your `shortcode_user_vars` dictionary. Also, the external variable will be written to `shortcode_user_vars`.

Supports the `_all_external` kwarg to retrieve all variables from an external .json file. Every key-value pair in the file will be stored to your `shortcode_user_vars` dictionary.

Supports the `_escape` parg to remove square brackets from the returned value. This is useful for when you want to use the result of `[get]` as a shortcode argument.

Supports the `_parse` parg to parse any shortcodes inside the returned value. This is useful when used in conjunction with `[set _defer]`. Note that global variables are parsed automatically. After parsing, the result is stored to the variable.

Supports the `_read_only` parg which is used in conjunction with `_parse` to prevent the variable from being overwritten by the parsed result.

```
My name is [get name]
```