Processes the first parg as either a `[function]` name or filepath, returning the result.

Functions take precedence over filepaths. You can declare a function with `[function some_method]` and execute it with `[call some_method]`.

As for filepaths, `unprompted/templates` is the base directory for this shortcode, e.g. `[call example/main]` will target `unprompted/templates/example/main.txt`.

If you do not enter a file extension, `.txt` is assumed.

Supports relative paths by starting the `path` with `./`, e.g. `[call ./main]` will target the folder that the previously-called `[call]` resides in.

This shortcode is powered by Python's glob module, which means it supports wildcards and other powerful syntax expressions. For example, if you wanted to process a random file inside of the `common` directory, you would do so like this: `[call common/*]`

You can set arbitrary user variables with kwargs, e.g. `[call roman_numeral_converter number=7]`.

The file is expected to be `utf-8` encoding. You can change this with the optional `_encoding` argument.

This shortcode is compatible with `[else]`. Here are the situations that will cause `[else]` to fire:

- The function has a `_required` argument that was not met.

- The filepath doesn't exist.

- Either the function or file return the term `_false`. (By the way, if this term is returned, it will not be printed.)

Supports the `_suppress_errors` parg to prevent writing errors to the console.

Supports the `_places` kwarg which lets you define multiple possible locations for the template you're calling. Places are prepended to the template path, and it will only return the first template that exists. Example: `[call "adjectives/something" _places="common|user"]` will check for `templates/common/adjectives/something` and `templates/user/adjectives/something`.




```
[call my_template/common/adjective]
```