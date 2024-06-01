Updates the content using argument pairings as replacement logic.

Arguments are case-sensitive.

Supports the optional `_from` and `_to` arguments, which can process secondary shortcode tags as replacement targets, e.g. `[replace _from="{get var_a}" _to="{get var_b}"]`. These arguments also support delimited values with `Unprompted.Config.syntax.delimiter`.

Supports the optional `_count` argument which limits the number of occurances to replace. For example, `[replace the="a" _count=1]the frog and the dog and the log[/replace]` will return `a frog and the dog and the log`.

Supports the optional `_insensitive` parg to enable case-insensitive search-and-replace.

Supports the optional `_load` kwarg for importing from:to replacement directions from one or more external JSON files.

Supports the `_now` parg to execute the replacement query before parsing the inner content.

Supports the `_strict` parg to only evaluate the `_to` expression on matches.

```
[replace red="purple" flowers="marbles"]
A photo of red flowers.
[/replace]
```
```
Result: A photo of purple marbles.
```