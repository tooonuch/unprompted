Checks whether `variable` is equal to the given value, returning the content repeatedly until the condition is false. This can create an infinite loop if you're not careful.

This shortcode also supports advanced expression syntax, e.g. `[while "some_var >= 5 and another_var < 2"]`. The following arguments are only relevant if you **don't** want to use advanced expressions:

Supports the testing of multiple variables, e.g. `[while var_a=1 var_b=50 var_c="something"]`. If one or more variables return false, the loop ends.

The optional `_any` argument will continue the loop if any of the provided conditions returns true.

The optional `_not` argument allows you to test for false instead of true, e.g. `[while _not my_variable=1]` will continue the loop so long as `my_variable` does *not* equal 1.

The optional `_is` argument allows you to specify the comparison logic for your arguments. Defaults to `==`, which simply checks for equality. Other options include `!=`, `>`, `>=`, `<` and `<=`. Example: `[while my_var="5" _is="<="]`

```
Advanced expression demo:
[set my_var]3[/set]
[while "my_var < 10"]
	Output
	[sets my_var="my_var + 1"]
[/while]
```

```
[set my_var]3[/set]
[while my_var="10" _is="<"]
	Output
	[sets my_var="my_var + 1"]
[/while]
```