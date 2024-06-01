Converts the verbs in the content to variety of conjugated forms.

Powered by the [pattern](https://github.com/clips/pattern/wiki/pattern-en) library - see pattern docs for more info.

Supports the optional `tense` argument. Defaults to `present`. Other options include: `infinitive`, `past`, `future`.

Supports the optional `person` argument for perspective. Defaults to `3`. Other options include: `1`, `2` and `none`.

Supports the optional `number` argument. Defaults to `singular`. Also supports `plural`.

Supports the optional `mood` argument. Defaults to `indicative`. Other options include: `imperative`, `conditional` and `subjunctive`.

Supports the optional `aspect` argument. Defaults to `imperfective`. Other options include: `perfective` and `progressive`.

Supports the optional `negated` boolean argument. Defaults to 0.

Supports the optional `parse` boolean argument. Defaults to 1.

Supports the optional `alias` argument, which is a shorthand "preset" for the above settings. Overrides your other arguments. The following aliases are supported: `inf`,`1sg`,`2sg`,`3sg`,`pl`,`part`,`p`,`1sgp`,`2sgp`,`3gp`,`ppl`,`ppart`.

```
[conjugate tense="past"]She says[/conjugate]
```
```
RESULT: She said
```