## Frequently Asked Questions

<details><summary>What is a "spice?"</summary>

A spice is a prompt template that uses a set of techniques to enhance the quality of the generated image. It can include anything from adding extra networks to using a negative prompt to using fluff terms.
</details>

<details><summary>Model compatibility</summary>

Spices are model-agnostic, meaning they are compatible with both Stable Diffusion 1.5 and SDXL checkpoints. Some settings such as the aspect ratio are automatically adjusted based on the architecture you're using.
</details>

<details><summary>Quality vs adherence</summary>

Optimizing for quality means that the model will try to generate the best possible image, even if it doesn't strictly adhere to the prompt. This can be useful for prompts that are too simple or too complex. However, if the spice strays too far from your intentions, try disabling GPT-2 prompt expansion and the use of negative prompts.
</details>