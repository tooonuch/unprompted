Allows you to exit a loop or `[function]` early.

## Supported shortcodes

The `[break]` statement has been tested inside of the following shortcodes:

- `[for]`
- `[while]`
- `[do]`
- `[each]`
- `[call]` (Both external files and functions)

## Example

Here is how you might use `[break]` inside of a `[for]` loop:

```
[for i=0 "i < 3" "i+1"]
	[log]Print: [get i][/log]
	[if "i > 1"]
		[break]
	[/if]
[/for]
[log]Script finished.[/log]
```

## Result

```
Print: 0
Print: 1
Script finished.
```
