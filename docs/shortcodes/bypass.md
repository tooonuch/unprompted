Allows you to disable the execution of specific shortcodes for the remainder of the run. It is similar to `[override]`, but for shortcodes instead of variables. Particularly useful for debugging purposes.

Provide the names of the shortcodes you wish to disable as pargs, separated by spaces.

If you supply `_toggle`, the shortcode can re-enable shortcodes that were previously bypassed.

```
[bypass repeat chance][repeat 3]do not print me[/repeat][chance 100]skip this too[/chance]print me
```
```
RESULT: print me
```