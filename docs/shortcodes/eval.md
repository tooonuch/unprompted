Parses the content using the simpleeval library, returning the result. Particularly useful for arithmetic.

simpleeval is designed to prevent the security risks of Python's stock `eval` function, however I make no assurances in this regard. If you wish to use Unprompted in a networked environment, do so at your own risk.

```
[eval]5 + 5[/eval]
```