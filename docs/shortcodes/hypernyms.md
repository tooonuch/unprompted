Replaces the content with one or more random hypernyms. This shortcode is powered by WordNet.

The optional `max` argument allows you to specify the maximum number of hypernyms to return. Defaults to -1, which returns all hypernyms. The hypernyms list is delimited by `Unprompted.Config.syntax.delimiter`.

```
[hypernyms max=1]dog[/hypernyms]
```

```
Possible result: animal
```