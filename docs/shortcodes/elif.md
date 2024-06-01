Shorthand "else if." Equivalent to `[else][if my_var="something"]content[/if][/else]`.

```
[set my_var]5[/set]
[if my_var=6]Discard this content[/if]
[elif my_var=5]Return this content![/elif]
```