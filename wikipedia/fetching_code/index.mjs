import { GoogleGenAI } from "@google/genai";
import { writeFile } from "fs/promises";

// models available on free tier:
// gemini-3-flash-preview | gemini-2.5-pro | gemini-2.5-flash | gemini-2.5-flash-preview-09-2025 | gemini-2.5-flash-lite | gemini-2.5-flash-lite-preview-09-2025 | gemini-2.0-flash | gemini-2.0-flash-lite
const MODEL = "gemini-2.5-flash-lite";

const SYS_PROMPT = `generate me a short poem based on the word i provide`;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const CALLS = 1;

export async function handler(event) {
  const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

  const ans = await Promise.all(
    Array.from({ length: CALLS }).map(async (_, i) => {
      const wikiArticle = await wikipedia_random_article();
      const writeWikiProm = writeArticle(wikiArticle, i, true);
      const sections = wikiArticle.map((sect) => sect.header);
      const aiArticle = await query_ai(ai, sections);
      await writeArticle(aiArticle, i, false);
      await writeWikiProm;
    }),
  );

  console.log(ans);
}

async function wikipedia_random_article() {
  const articleName = (
    await fetch_wikipedia({
      action: "query",
      format: "json",
      list: "random",
      rnnamespace: 0,
      rnlimit: 1,
    })
  ).query.random[0].title;
  const articlePages = (
    await fetch_wikipedia({
      action: "query",
      prop: "extracts", //no metadata
      explaintext: 1, //plaintext, no html
      titles: articleName,
      format: "json",
    })
  ).query.pages;
  const article =
    Object.keys(articleName).length > 0
      ? articlePages[Object.keys(articlePages)[0]].extract
      : "";
  const parsedArticle = parse_sections(article, articleName);
  return parsedArticle;
}

async function fetch_wikipedia(params) {
  const endpoint = "https://en.wikipedia.org/w/api.php";

  const paramStr = new URLSearchParams(params).toString();
  const resp = await fetch(endpoint + "?" + paramStr);
  const text = await resp.text();
  return JSON.parse(text);
}

/**
 *
 * @param {string} rawtext
 */
function parse_sections(rawtext, title) {
  const lines = rawtext.split(/\r?\n/).map((x) => x.trim());
  const sections = [];

  let lastTitle = -1;
  let lastHeader = title;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("==") && lines[i].endsWith("==")) {
      // some type of heading
      const lastSection = lines.slice(lastTitle + 1, i).join("\n");
      sections.push({ header: lastHeader, text: lastSection });
      lastHeader = lines[i].replace(/=/g, " ").trim();
      lastTitle = i;
    }
  }
  const lastSection = lines.slice(lastTitle + 1).join("");
  sections.push({ header: lastHeader, text: lastSection });
  return sections;
}

async function query_ai(ai, sections, id) {
  console.log(`Title: ${sections[0]}\nSections:${JSON.stringify(sections)}`);
  //return [{ header: "X", text: "AAAAA" }];
  try {
    const resp = await ai.models.generateContent({
      model: MODEL,
      contents: `Title: ${sections[0]}\nSections:${JSON.stringify(sections)}`,
      config: { systemInstruction: SYS_PROMPT },
    });
    console.log(`AI Request #${id} succeeded`);
    await writeFile("AIOUT.txt", resp.text);
    return parse_sections(resp.text);
  } catch (err) {
    console.log(`AI Request #${id} failed`);
    console.log(err);
    return {};
  }
}

async function writeArticle(parsed, index, isWikipedia) {
  if (parsed.length == 0) {
    await writeFile(
      `data/${isWikipedia ? "wiki" : "ai"}-${index}.FAILED.txt`,
      "FAILED",
    );
  }
  const title = parsed[0].header;
  for (let section of parsed) {
    await writeFile(
      `data/${isWikipedia ? "wiki" : "ai"}-${index}-${title}.${section.header}.txt`,
      section.text,
    );
  }
}

await handler();
