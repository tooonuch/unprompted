Checks whether `variable` is equal to the given value, returning the content if true, otherwise discarding the content.

Supports the testing of multiple variables, e.g. `[if var_a=1 var_b=50 var_c="something"]`. If one or more variables return false, the content is discarded.

The optional `_any` argument allows you to return the content if one of many variables return true. This is the equivalent of running "or" instead of "and" in programming, e.g. `[if _any var_a=1 var_b=50]`.

The optional `_not` argument allows you to test for false instead of true, e.g. `[if _not my_variable=1]` will return the content if `my_variable` does *not* equal 1.

The optional `_is` argument allows you to specify the comparison logic for your arguments. Defaults to `==`, which simply checks for equality. Other options include `!=`, `>`, `>=`, `<` and `<=`. Example: `[if my_var="5" _is="<="]`

Supports [advanced expressions](#advanced-expressions) - useful for testing complex conditions.

```
[if subject="man"]wearing a business suit[/if]
```

```
(Advanced expression demo)
[if "subject is 'man' or subject is 'woman'"]wearing a shirt[/if]
```