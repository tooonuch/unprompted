Prints metadata about the content. You must pass the type(s) of data as positional arguments.

Supports `character_count` for retrieving the number of individual characters in the content.

Supports `word_count` for retrieving the number of words in the content, using space as a delimiter.

Supports `sentence_count` for retrieving the number of sentences in the content. Powered by the nltk library.

Supports `filename` for retrieving the base name of a file from the filepath content. For example, if the content is `C:/pictures/delicious hamburger.png` then this will return `delicious hamburger`.

Supports `string_count` for retrieving the number of a custom substring in the content. For example, `[info string_count="the"]the frog and the dog and the log[/info]` will return 3.

Supports `clip_count` for retrieving the number of CLIP tokens in the content (i.e. a metric for prompt complexity.) This argument is only supported within the A1111 WebUI environment.

```
[info word_count]A photo of Emma Watson.[/info]
```
```
Result: 5
```