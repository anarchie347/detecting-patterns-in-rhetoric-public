import { writeFileSync } from "fs";

writeFileSync("out.json", JSON.stringify(parse_sections(article), null, 2));

/*
 in the style of the Wittgenstein language game of Bullshit, write the


following wikipedia article given the title and headings. Format your

writing as if it was taken from a wikimedia API request with

explaintext=1 to produce only raw text, such as using formatting

== Heading == for headings. DO not include any JSON, only the page

extract. Do not include any preamble or postamble about your response.

The wikipedia article is:

Title: Breadth-first search

Headings: Psuedocode, Analysis, BFS ordering, Applications 
*/
