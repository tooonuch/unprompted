Replaces the content with one or more random synonyms. This shortcode is powered by a combination of WordNet and Moby Thesaurus II. Does not require an online connection after first use (word databases are downloaded to disk.)

The optional `max` argument allows you to specify the maximum number of synonyms to return. Defaults to -1, which returns all synonyms. The synonym list is delimited by `Unprompted.Config.syntax.delimiter`.

The optional `include_self` positional argument determines whether the original content can be returned as a possible result. Defaults to False.

The optional `enable_moby` keyword argument determines whether Moby Thesaurus II will be referenced. Defaults to True. On first use, the Moby Thesaurus will be downloaded to the `lib_unprompted` folder - it is about 24 MB.

The optional `enable_wordnet` keyword argument determines whether WordNet will be referenced. Defaults to True.

```
[synonyms]amazing[/synonyms]
```