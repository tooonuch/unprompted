Randomly returns one of multiple options, as delimited by the vertical pipe or newline character.

Supports `_case` which overrides the random nature of this shortcode with a pre-determined index (starting at 0.) Example: `[choose _case=1]red|yellow|green[/choose]` will always return `yellow`. You can also pass a variable into this argument.

Supports an optional positional argument that tells the shortcode how many times to execute (default 1). For example: `[choose 2]Artist One|Artist Two|Artist Three|Artist Four[/choose]` will return two random artists.

Supports the `_allow_dupe` parg which allows the same option to be returned multiple times. For example: `[choose 2 _allow_dupe]Artist One|Artist Two|Artist Three|Artist Four[/choose]` might return `Artist One, Artist One`.

Supports the optional `_sep` argument which is a string delimiter that separates multiple options to be returned (defaults to `, `). In the example above, you might get `Artist One, Artist Three` as a result. When only returning one option, `_sep` is irrelevant.

Supports the optional `_weighted` argument, which allows you to customize the probability of each option. Weighted mode expects the content to alternate between **weight value** and **the option itself** using the standard delimiter. For example, if you want your list to return Apple 30% of the time, Strawberry 50% of the time, and Blueberry 20% of the time you can do it like this:

```
[choose _weighted]
3|Apple
5|Strawberry
2|Blueberry
[/choose]
```

If you skip a weight value--e.g. `3|Apple|Strawberry`--then the following option (Strawberry) will automatically have a weight value of 1.

The weight value dictates the number of times that an option is added to the master list of choices, which is then shuffled and picked from at random. So, if your content is `2|Blue|3|Red|Green` the master list becomes `Blue,Blue,Red,Red,Red,Green`.

Note that if you wish to include the standard delimiter as a literal character in a choice, you can use the macro string `%PIPE%` to represent it. For example `[choose]something|another%PIPE%thing[/choose]` may return `another|thing`.

Supports the `_raw` parg, which prevents the execution of inner shortcodes except the one that is selected by `[choose]`.

```
[choose]red|yellow|blue|green[/choose]
```