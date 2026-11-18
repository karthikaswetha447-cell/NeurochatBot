
function getClassification() {
  const headings = document.querySelectorAll("h3");

  for (const heading of headings) {
    if (!heading.innerText.includes("Student Message Classification")) {
      continue;
    }

    let element = heading;

    // Search parent elements for the classification details.
    for (let i = 0; i < 8 && element; i++) {
      const text = element.innerText || "";

      const categoryMatch = text.match(
        /Category\s*:\s*(Admission Enquiry|Course Information|Payment Issue|Certificate Issue|Appreciation|Complaint|Other)/i
      );

      if (categoryMatch) {
        const reasonMatch = text.match(
          /Reason\s*:\s*([\s\S]*?)(?:\n|Suggested Action|$)/i
        );

        return {
          category: categoryMatch[1],
          reason: reasonMatch
            ? reasonMatch[1].trim()
            : "Classification provided by NeuroBot."
        };
      }

      element = element.parentElement;
    }
  }

  return null;
}
