/**
 * Give a badge on Unprompted Accordion indicating total number of active 
 * wizard scripts.
*/
(function()
{
	const unprompted_accordions = new Set();
	const dropdown_prefix = "✓ ";

	function update_active_unit_count(accordion)
	{
		const span = accordion.querySelector(".label-wrap span");
		let active_unit_count = 0;
		accordion.querySelectorAll(".wizard-autoinclude input").forEach(function(checkbox)
		{
			if (checkbox.checked)
			{
				active_unit_count++;
			}
		});

		if (span.childNodes.length !== 1)
		{
			span.removeChild(span.lastChild);
		}
		if (active_unit_count > 0)
		{
			const div = document.createElement("div");
			div.classList.add("unprompted-badge");
			div.innerHTML = `🪄 ${active_unit_count} wizard${active_unit_count > 1 ? "s" : ""}`;
			span.appendChild(div);
		}
	}

	/* function on  */
	function clear_wizards(accordion, mode)
	{
		// Are you sure? dialog
		if (!confirm("Are you sure you want to clear all active wizards?")) return;

		const wizard_checkboxes = accordion.querySelectorAll(`${mode} .wizard-autoinclude input`);
		wizard_checkboxes.forEach(function(checkbox)
		{
			checkbox.checked = false;
			checkbox.dispatchEvent(new Event("change"));
		});

		// Update badge counts
		update_active_unit_count(accordion);

		// Clear autoincludes
		accordion.unprompted_autoincludes.clear();
	}

	onUiUpdate(() =>
	{
		gradioApp().querySelectorAll(".unprompted-accordion").forEach(function(accordion)
		{
			// Gradio dropdowns are created dynamically, so we must add the autoinclude class whenever one is created
			accordion.querySelectorAll("#wizard-dropdown").forEach(function(dropdown)
			{
				var input = dropdown.querySelector("input");
				input.classList.remove("autoincluded");

				dropdown.querySelectorAll("li").forEach(function(li)
				{
					const content = li.textContent.trim();

					if (accordion.unprompted_autoincludes.has(content))
					{
						li.classList.add("autoincluded");
					}
				});

				if (accordion.unprompted_autoincludes && accordion.unprompted_autoincludes.has(dropdown_prefix + input.value))
				{
					input.classList.add("autoincluded");
				}
			});

			// Startup
			if (unprompted_accordions.has(accordion)) return;

			accordion.unprompted_autoincludes = new Set();
			unprompted_accordions.add(accordion);

			function update_autoincludes(checkbox)
			{
				const affix = checkbox.nextElementSibling.textContent.replace("🪄 Auto-include ", "").replace(" in:", "").replace("[", "").replace("]", "").trim();
				const template_name = dropdown_prefix + affix;

				// Add or remove template_name from unprompted_autoincludes
				accordion.unprompted_autoincludes.has(template_name) ? accordion.unprompted_autoincludes.delete(template_name) : accordion.unprompted_autoincludes.add(template_name);
			}

			accordion.querySelectorAll(".wizard-autoinclude input").forEach(function(checkbox)
			{
				checkbox.addEventListener("change", function()
				{
					update_active_unit_count(accordion);
					update_autoincludes(checkbox);
				});

				// Check if checkbox is checked
				if (checkbox.checked)
				{
					update_autoincludes(checkbox);
				}
			});

			/* click #templates-clear or #shortcodes-clear */
			accordion.querySelectorAll("#templates-clear, #shortcodes-clear").forEach(function(button)
			{
				button.addEventListener("click", function()
				{
					// Check if button is a descendant of #wizard-templates
					if (button.closest("#wizard-templates"))
					{
						var mode = "#wizard-templates";
					}
					else
					{
						var mode = "#wizard-shortcodes";
					}

					clear_wizards(accordion, mode);
				});
			});

			// Perform count on startup
			update_active_unit_count(accordion);

		});
	});


})();