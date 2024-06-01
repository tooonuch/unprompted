Returns a delimited string containing the full paths of all files in a given path or paths (separate multiple paths with the standard delimiter.)

This shortcode is powered by Python's glob module, which means it supports wildcards and other powerful syntax expressions.

Supports the optional `_delimiter` argument which lets you specify the separator between each filepath. It defaults to your config's `syntax.delimiter` value (`|`).

Supports the optional `_basename` parg that causes the shortcode to return filenames instead of full paths.

Supports the optional `_hide_ext` parg that causes the shortcode to trim filename extensions out of the returned string.

Supports the macro `%BASE_DIR%` which will be substituted with an absolute path to the Unprompted extension.

Supports the `_recursive` parg to search within subfolders of the specified path.

Supports the `_places` kwarg which is a delimited list of path parts to check. It replaces the `%PLACE%` macro string in your main path. Example: `[filelist "C:/some/%PLACE%/pictures/*.*" _places="one|two"]` will check for files in `C:/some/one/pictures` and `C:/some/two/pictures`.

```
[filelist "C:/my_pictures/*.*"]
```