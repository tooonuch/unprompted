Do-until style loop. The content is processed, then the `until` expression is evaluated - if it's false, the content is processed again. Repeat until `until` is true.

```
[sets my_var=0]
[do until="my_var > 5"]
	Print me
	[sets my_var="my_var + 1"]
[/do]
```